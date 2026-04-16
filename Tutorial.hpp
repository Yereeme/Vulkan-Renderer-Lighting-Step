#pragma once

#include "PosColVertex.hpp"
#include "PosNorTexVertex.hpp"
#include "mat4.hpp"
#include <string>
#include <algorithm>
#include <optional>
#include "frame_time_logger.hpp"
 

#include "S72.hpp"
#include "RTG.hpp"
#include <unordered_map>
#include <cstdint>

struct Tutorial : RTG::Application {

	//Tutorial(RTG &);
	Tutorial(RTG& rtg, std::string const& scene_file, RTG::Configuration::CullingMode culling_mode);
	Tutorial(Tutorial const &) = delete; //you shouldn't be copying this object
	~Tutorial();

	 
	 
	//kept for use in destructor:
	RTG &rtg;
	std::string scene_file;
	S72 scene; //loaded s72 scene graph  
	bool use_s72_scene = true;
	std::vector<uint32_t> visible_instances;
	//FrameTimeLogger frame_logger;
	
	//bool ft_log_enabled = false;
	// camera override for headless (optional)

	 
	


 
	//void create_descriptor_pool(RTG& rtg);

	//--------------------------------------------------------------------
	//Resources that last the lifetime of the application:

	//chosen format for depth buffer:
	VkFormat depth_format{};
	//Render passes describe how pipelines write to images:
	VkRenderPass render_pass = VK_NULL_HANDLE;

	VkRenderPass shadow_render_pass = VK_NULL_HANDLE;

	//Pipelines:
	//...
	//adding BackgroundPipeline member structure to Tutorial.cppp

	struct BackgroundPipeline {
		//descriptor set layouts
		VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;
		VkDescriptorSet set = VK_NULL_HANDLE; // allocated from your global pool (or a small one) 
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
		// push constants
		struct Push {
			float time; //4 bytes
			float _pad0[3]; //16
			mat4 inv_view_rot; // inverse view rotation only (no translation) (64)
			mat4 inv_proj; //(64)
			float exposure;   // 2^exposure
			int tone_op;      // 0 linear, 1 reinhard
			int _pad1[2];     // pad to 16 bytes
		};
		static_assert(sizeof(Tutorial::BackgroundPipeline::Push) == 160, "Push must match GLSL layout.");

		//no vertex bindings

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} background_pipeline;
	//...
	//LinesPipeline for grid
	struct LinesPipeline {
		 
		//descriptor set layouts:
		VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;

		//types for descriptors:
		struct Camera {
			mat4 CLIP_FROM_WORLD;
		};
		static_assert(sizeof(Camera) == 16 * 4, "Camera buffer structure is packed");
		// push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//we have vertex bindings now!
		using Vertex = PosColVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG&, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG&);
	} lines_pipeline;

	//ObjectsPipeline  
	struct ObjectsPipeline {

		//descriptor set layouts:
 
		VkDescriptorSetLayout set0_World = VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;
		VkDescriptorSetLayout set3_EnvLambertian = VK_NULL_HANDLE;
		VkDescriptorSetLayout set4_Lights = VK_NULL_HANDLE;
		VkDescriptorSetLayout set5_Shadow = VK_NULL_HANDLE;

		struct World {
			struct { float x, y, z, padding_; } SKY_DIRECTION;
			struct { float r, g, b, padding_; } SKY_ENERGY;
			struct { float x, y, z, padding_; } SUN_DIRECTION;
			struct { float r, g, b, padding_; } SUN_ENERGY;
		};
		static_assert(sizeof(World) == 4 * 4 + 4 * 4 + 4 * 4 + 4 * 4, "World is the expected size.");

		struct Push {
			mat4 LIGHT_CLIP_FROM_WORLD;
			int32_t SHADOW_LIGHT_INDEX;
			int32_t _pad[3];
		};
		static_assert(sizeof(Push) == 80, "Objects Push must be 80 bytes.");

		//using Camera = LinesPipeline::Camera;
		struct Transform { //storage buffer descriptor set layout
			mat4 CLIP_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL_NORMAL;

		};
		static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4, "Transform is the expected size.");
	 
 
		//push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//we have vertex bindings 

		using Vertex = PosNorTexVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG& rtg, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG& rtg);
	} objects_pipeline;

	struct MirrorPipeline {

		// Mirror does NOT own descriptor set layouts.
		// It reuses:
		//   set 0 layout from BackgroundPipeline (samplerCube envTex)
		//   set 1 layout from ObjectsPipeline (Transforms SSBO)

		// --- descriptor types (optional, keep if you use it elsewhere) ---
		struct Transform {
			mat4 CLIP_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL_NORMAL;
		};
		static_assert(sizeof(Transform) == 16 * 4 + 16 * 4 + 16 * 4, "Transform is the expected size.");

		// --- push constants ---
		 
		struct Push {
			S72::vec3 camera_ws; // 12
			float exposure;      // 4  -> 16

			int32_t tone_op;     // 4
			int32_t _pad[3];     // 12 -> 32
		};
		static_assert(sizeof(Push) == 32, "Push must match GLSL layout.");

		// --- pipeline ---
		VkPipelineLayout layout = VK_NULL_HANDLE;
		using Vertex = PosNorTexVertex;
		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG& rtg,
			VkRenderPass render_pass,
			uint32_t subpass,
			VkDescriptorSetLayout background_set0_env_layout,
			VkDescriptorSetLayout objects_set1_transforms_layout
			);

		void destroy(RTG& rtg);

	} mirror_pipeline;

	//PBRPipeline
	struct PBRPipeline {

		//descriptor set layouts:
		VkDescriptorSetLayout set0_World = VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;
		VkDescriptorSetLayout set3_EnvPBR = VK_NULL_HANDLE;
		VkDescriptorSetLayout set4_Lights = VK_NULL_HANDLE;
		VkDescriptorSetLayout set5_Shadow = VK_NULL_HANDLE;

		struct Push {
			mat4 CLIP_FROM_LOCAL;        // 64
			mat4 WORLD_FROM_LOCAL;       // 64
			mat4 LIGHT_CLIP_FROM_WORLD;  // 64

			S72::vec3 camera_ws;         // 12
			float exposure;              // 4  -> 16

			int32_t tone_op;             // 4
			int32_t SHADOW_LIGHT_INDEX;  // 4
			float _padding[2];           // 8  -> tail = 16
		};
		static_assert(sizeof(Push) == 224, "PBR Push must be 224 bytes!");

		//types for descriptors: (i reuse object pipeline ones)
		using World = ObjectsPipeline::World;
		using Transform = ObjectsPipeline::Transform;


		//no push constants
		VkPipelineLayout layout = VK_NULL_HANDLE;

		//we have vertex bindings 
		using Vertex = PosNorTexVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG& rtg, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG& rtg);
	} pbr_pipeline;

	//pools from which per-workspace things are allocated:
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

	//workspaces hold per-render resources:
	struct Workspace {
		VkCommandBuffer command_buffer = VK_NULL_HANDLE; //from the command pool above; reset at the start of every render.

		//location for lines data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer lines_vertices_src; //host coherent; mapped
		Helpers::AllocatedBuffer lines_vertices; //device-local

		//location for LinesPipeline::Camera data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Camera_src; //host coherent; mapped
		Helpers::AllocatedBuffer Camera; //device-local
		VkDescriptorSet Camera_descriptors; //references Camera

		//location for ObjectsPipeline::World data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer World_src; //host coherent; mapped
		Helpers::AllocatedBuffer World; //device-local
		VkDescriptorSet World_descriptors; //references World

		//location for ObjectsPipeline::Transforms data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Transforms_src; //host coherent; mapped
		Helpers::AllocatedBuffer Transforms; //device-local
		VkDescriptorSet Transforms_descriptors; //references Tranforms

		//gpu location for lights data 
		Helpers::AllocatedBuffer Lights; //host coherent; mapped
		VkDescriptorSet Lights_descriptors = VK_NULL_HANDLE;

		VkDescriptorSet PBR_Env_descriptors = VK_NULL_HANDLE;

		VkDescriptorSet Shadow_descriptors = VK_NULL_HANDLE;

		std::vector<VkDescriptorSet> Shadow_descriptors_per_light;

	 


	};
	std::vector< Workspace > workspaces;

	//-------------------------------------------------------------------
	//static scene resources:

	Helpers::AllocatedBuffer object_vertices;

	

	//store the index of the first vertex and the count of vertices (parameters used by vkCmdDraw) for each 
	//mesh stored in obj vertices arraay
	struct ObjectVertices {
		uint32_t first = 0;
		uint32_t count = 0;

		//local - space AABB for this mesh range :
		S72::vec3 local_min{ +INFINITY, +INFINITY, +INFINITY }; //for each vertex position p update min
		S72::vec3 local_max{ -INFINITY, -INFINITY, -INFINITY };


	};

	//--- S72 packed mesh ranges (one entry per scene.meshes[i]):
	std::vector< ObjectVertices > s72_mesh_vertices;
	std::unordered_map<S72::Mesh const*, ObjectVertices> s72_mesh_to_range;

	

	ObjectVertices plane_vertices;
	ObjectVertices torus_vertices;
	ObjectVertices crystal_vertices;
	ObjectVertices chen_sword_vertices;
	ObjectVertices chen_body_vertices;
	ObjectVertices chen_clothes_vertices;
	ObjectVertices chen_face_vertices;
	ObjectVertices chen_hairs_vertices;
	ObjectVertices chen_iris_vertices;

	//uint32_t character_texture = 0;
	uint32_t env_texture = 0; // index into `textures` / `texture_descriptors`
	bool has_env_texture = false;
	Helpers::AllocatedImage env_cubemap;
	VkImageView env_cubemap_view = VK_NULL_HANDLE;
	std::vector< Helpers::AllocatedImage > textures;
	std::vector< VkImageView > texture_views;
	std::vector<Helpers::AllocatedImage> normal_maps;
	std::vector<VkImageView> normal_map_views;
	std::unordered_map<std::string, uint32_t> normal_path_to_index;
	VkSampler texture_sampler = VK_NULL_HANDLE;
	std::unordered_map<std::string, uint32_t> material_name_to_set2; // material name -> set2 index
	std::vector<VkDescriptorSet> material_descriptors;               // the actual set2s for S72 materials
	// set2 descriptor sets per material:
// - lambertian pipeline expects 2 bindings (albedo, normal)
// - pbr pipeline expects 4 bindings (albedo, normal, roughness, metalness)
	std::vector<VkDescriptorSet> material_descriptors_lam;
	std::vector<VkDescriptorSet> material_descriptors_pbr;

	// per-material indices into textures/normal_maps arrays:
	std::vector<uint32_t> roughness_idx_per_set2;
	std::vector<uint32_t> metalness_idx_per_set2;
	VkSampler env_sampler = VK_NULL_HANDLE;
	VkDescriptorPool texture_descriptor_pool = VK_NULL_HANDLE;
	std::vector< VkDescriptorSet > texture_descriptors; //allocated from texture_descriptor_pool
	//maps a loaded material texture to our textures[] index:
	std::unordered_map<std::string, uint32_t> texture_lookup;//cpu side metadata
	std::unordered_map< S72::Material const*, uint32_t > material_to_texture;
	std::unordered_map< std::string, uint32_t > s72_texture_path_to_index; // cache

	uint32_t tex_body = 0;
	uint32_t tex_clothes = 0;
	uint32_t tex_hair = 0;
	uint32_t tex_face = 0;
	uint32_t tex_iris = 0;
	uint32_t tex_sword = 0;

	//--------------------------------------------------------------------
	//Resources that change when the swapchain is resized:

	virtual void on_swapchain(RTG &, RTG::SwapchainEvent const &) override;

	Helpers::AllocatedImage swapchain_depth_image;
	VkImageView swapchain_depth_image_view = VK_NULL_HANDLE;
	std::vector< VkFramebuffer > swapchain_framebuffers;

	//used from on_swapchain and the destructor: (framebuffers are created in on_swapchain)
	void destroy_framebuffers();

	//shadow map size and stuff members
	std::vector<Helpers::AllocatedImage> shadow_maps;; //GPU image (texture) that stores depth
	std::vector<VkImageView> shadow_map_views;
	std::vector<VkFramebuffer> shadow_framebuffers;
	
	 

	//--------------------------------------------------------------------
	//Resources that change when time passes or the user interacts:

	virtual void update(float dt) override;
	virtual void on_input(InputEvent const &) override;

	//modal action, interceps inputs:
	std::function< void(InputEvent const&) > action;

		//global variable
	float time = 0.0f;

	struct OrbitCamera {
		float target_x = 0.0f, target_y = 0.0f, target_z = 0.0f; //where the camera is 
		//looking + orbiting
		float radius = 2.0f; //distance from camera to target
		float azimuth = 0.0f; //counterclockwise angle around z axis between x axis and camera direction
		//(radians)
		float elevation = 0.25f * float(M_PI); //angle up from xy plane to camera direction (radians)

		float fov = 60.0f / 180.0f * float(M_PI); //vertical field of view (radians)
		float near = 0.1f; //near clipping plane
		float far = 1000.0f; //far clipping plane
	} free_camera;

	//for selecting between cameras:
	enum class CameraMode {
		Scene = 0,
		User = 1,
		Debug = 2,
	} camera_mode = CameraMode::User;

	std::optional<Tutorial::CameraMode> forced_camera;


	//render-time viewport/scissor (computed in update, applied in render)
	VkViewport draw_viewport{};
	VkRect2D   draw_scissor{};

	//scene camera aspect (used to letterbox/pillarbox)
	float scene_cam_aspect = 0.0f; //fullscreen

	void compute_letterbox(float target_aspect);



	// Scene camera list (nodes that have a camera attached):
	std::vector<S72::Node const*> scene_camera_nodes;
	uint32_t active_scene_camera = 0;

	// second user-controlled camera for debug mode:
	OrbitCamera debug_camera;

	// store the culling camera matrix
	mat4 CLIP_FROM_CULL = mat4{}; // set every update
	bool debug_cull_locked = false;
	mat4 debug_locked_CLIP_FROM_CULL = mat4{};

	// --- A2-diffuse: lambertian irradiance cubemap ---
	Helpers::AllocatedImage env_lambertian_cubemap;
	VkImageView env_lambertian_cubemap_view = VK_NULL_HANDLE;
	VkDescriptorSet env_lambertian_descriptors = VK_NULL_HANDLE; // set=3 for ObjectsPipeline
 
	bool has_env_lambertian = false;

	// --- A2-pbr: env resources ---
	Helpers::AllocatedImage env_ggx_cubemap;
	VkImageView env_ggx_cubemap_view = VK_NULL_HANDLE;
	bool has_env_ggx = false;

	// --- A2-pbr: BRDF LUT (2D) ---
	Helpers::AllocatedImage brdf_lut;
	VkImageView brdf_lut_view = VK_NULL_HANDLE;
	bool has_brdf_lut = false;

	VkDescriptorSet env_pbr_descriptors = VK_NULL_HANDLE; // set=3 for PBRPipeline

	Helpers::AllocatedImage dummy_brdf_lut;
	VkImageView dummy_brdf_lut_view = VK_NULL_HANDLE;


	//computed from the current camera (as set by camera_mode) during update():
	mat4 CLIP_FROM_WORLD; //matrix through which to view grid line
	mat4 VIEW_FROM_WORLD;
	mat4 CLIP_FROM_VIEW;
	std::vector< LinesPipeline::Vertex > lines_vertices;

	ObjectsPipeline::World world;

	struct LoadedLight {
		enum class Type {
			Sun,
			Sphere,
			Spot
		};

		Type type = Type::Sun;

		std::string name;

		S72::vec3 tint = { 1.0f, 1.0f, 1.0f };
		S72::vec3 world_position = { 0.0f, 0.0f, 0.0f };
		S72::vec3 world_direction = { 0.0f, 0.0f, -1.0f };
		float shadow = 0.0f;

		mat4 world_from_local = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};

		float radius = 0.0f;
		float power = 0.0f;
		float limit = 0.0f;

		float angle = 0.0f;
		float strength = 0.0f;

		float fov = 0.0f;
		float blend = 0.0f;
	};

	std::vector<LoadedLight> loaded_lights;
	std::vector<LoadedLight*> shadow_spot_lights;
	VkSampler shadow_sampler = VK_NULL_HANDLE;

	struct GPULight {
		vec4 position;
		vec4 direction;
		vec4 tint;
		vec4 params;
	};

	struct ShadowPush {
		mat4 LIGHT_CLIP_FROM_WORLD;
		int32_t OBJECT_INDEX;
		int32_t _pad[3];
	};
	static_assert(sizeof(ShadowPush) == 80, "ShadowPush must be 80.");

	struct ShadowPipeline {
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;

		VkPipeline handle = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;

		void create(RTG& rtg, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG& rtg);
	} shadow_pipeline;

	struct ObjectInstance {
		ObjectVertices vertices;
		ObjectsPipeline::Transform transform;
		uint32_t texture = 0;
		S72::Material const* material = nullptr;

	};
	std::vector< ObjectInstance > object_instances;
	std::vector<uint32_t> mirror_instance_indices;
	std::vector<uint32_t> pbr_instance_indices;
	// per-set2 texture indices (debug / convenience):
	std::vector<uint32_t> albedo_idx_per_set2;
	std::vector<uint32_t> normal_idx_per_set2;



	//visible list
	std::vector<ObjectInstance> visible_object_instances;
	RTG::Configuration::CullingMode culling_mode = RTG::Configuration::CullingMode::None;

	bool enable_culling = false; //toggle
	//--------------------------------------------------------------------
	//Rendering function, uses all the resources above to queue work to draw a frame:

	virtual void render(RTG &, RTG::RenderParams const &) override;

	//   animation state  
	bool anim_started = false;
	float anim_time = 0.0f;
	bool anim_paused = false;

	
};
