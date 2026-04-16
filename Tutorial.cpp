#include "Tutorial.hpp"

#include "VK.hpp"
#include "LightHelpers.hpp"
#include <GLFW/glfw3.h>
#include <variant>
#include <vector>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <filesystem>


 
#include "external\tinyobjloader\tiny_obj_loader.h"

 
#include "external\tinyobjloader\stb_image.h"

static mat4 mat4_transpose(mat4 const& m) {
	return mat4{
		m[0], m[4], m[8],  m[12],
		m[1], m[5], m[9],  m[13],
		m[2], m[6], m[10], m[14],
		m[3], m[7], m[11], m[15]
	};
}

 




static void upload_cubemap_faces_rgba8(
	RTG& rtg,
	VkCommandPool command_pool,
	Helpers::AllocatedImage const& cubemap,
	int faceSize,
	std::array<std::vector<uint8_t>, 6> const& faces
) {
	assert(cubemap.handle != VK_NULL_HANDLE);
	assert(faceSize > 0);

	VkDeviceSize faceBytes = VkDeviceSize(faceSize) * VkDeviceSize(faceSize) * 4;

	auto staging = rtg.helpers.create_buffer(
		size_t(faceBytes * 6),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		Helpers::Mapped
	);

	uint8_t* dst = reinterpret_cast<uint8_t*>(staging.allocation.data());
	for (int face = 0; face < 6; ++face) {
		assert(faces[face].size() == size_t(faceBytes));
		std::memcpy(dst + faceBytes * face, faces[face].data(), size_t(faceBytes));
	}

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	{
		VkCommandBufferAllocateInfo alloc{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VK(vkAllocateCommandBuffers(rtg.device, &alloc, &cmd));

		VkCommandBufferBeginInfo begin{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK(vkBeginCommandBuffer(cmd, &begin));
	}

	{
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = cubemap.handle,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 6,
			},
		};

		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
	}

	{
		std::array<VkBufferImageCopy, 6> regions{};
		for (int face = 0; face < 6; ++face) {
			regions[face] = VkBufferImageCopy{
				.bufferOffset = VkDeviceSize(face) * faceBytes,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = uint32_t(face),
					.layerCount = 1,
				},
				.imageOffset{0, 0, 0},
				.imageExtent{uint32_t(faceSize), uint32_t(faceSize), 1},
			};
		}

		vkCmdCopyBufferToImage(
			cmd,
			staging.handle,
			cubemap.handle,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			uint32_t(regions.size()),
			regions.data()
		);
	}

	{
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = cubemap.handle,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 6,
			},
		};

		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
	}

	VK(vkEndCommandBuffer(cmd));

	{
		VkSubmitInfo submit{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd,
		};
		VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit, VK_NULL_HANDLE));
		VK(vkQueueWaitIdle(rtg.graphics_queue));
	}

	vkFreeCommandBuffers(rtg.device, command_pool, 1, &cmd);
	rtg.helpers.destroy_buffer(std::move(staging));
}
static void upload_cubemap_faces_float4_mip(
	RTG& rtg,
	Helpers::AllocatedImage const& cubemap,
	int faceSize,
	uint32_t mipLevel,
	std::array<std::vector<float>, 6> const& faces
) {
	assert(cubemap.handle != VK_NULL_HANDLE);

	VkDeviceSize faceBytes = VkDeviceSize(faceSize) * VkDeviceSize(faceSize) * 4 * sizeof(float);

	// Create staging buffer
	auto staging = rtg.helpers.create_buffer(
		size_t(faceBytes * 6),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		Helpers::Mapped
	);

	uint8_t* dst = reinterpret_cast<uint8_t*>(staging.allocation.data());
	for (int face = 0; face < 6; ++face) {
		std::memcpy(dst + faceBytes * face, faces[face].data(), size_t(faceBytes));
	}

	// 1. Manually allocate a command buffer
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkCommandBufferAllocateInfo alloc_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = rtg.helpers.transfer_command_pool, // Use the pool your code already uses
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &cmd));

	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	VK(vkBeginCommandBuffer(cmd, &begin_info));

	// 2. Transition specific mip to TRANSFER_DST
	VkImageMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.image = cubemap.handle,
		.subresourceRange{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = mipLevel,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 6,
		},
	};

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	// 3. Copy buffer to image
	std::array<VkBufferImageCopy, 6> regions{};
	for (int face = 0; face < 6; ++face) {
		regions[face] = {
			.bufferOffset = VkDeviceSize(face) * faceBytes,
			.imageSubresource{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = mipLevel,
				.baseArrayLayer = uint32_t(face),
				.layerCount = 1,
			},
			.imageExtent = { uint32_t(faceSize), uint32_t(faceSize), 1 }
		};
	}
	vkCmdCopyBufferToImage(cmd, staging.handle, cubemap.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions.data());

	// 4. Transition to SHADER_READ_ONLY
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	VK(vkEndCommandBuffer(cmd));

	// 5. Submit and wait
	VkSubmitInfo submit_info{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, VK_NULL_HANDLE));
	VK(vkQueueWaitIdle(rtg.graphics_queue));

	vkFreeCommandBuffers(rtg.device, rtg.helpers.transfer_command_pool, 1, &cmd);
	rtg.helpers.destroy_buffer(std::move(staging));
}
static void upload_cubemap_faces_float4(
	RTG& rtg,
	Helpers::AllocatedImage const& cubemap,
	int faceSize,
	std::array<std::vector<float>, 6> const& faces
) {
	// just mip 0 upload
	upload_cubemap_faces_float4_mip(rtg, cubemap, faceSize, 0, faces);
}



[[maybe_unused]]  static bool split_cube_faces_vertical_strip_u8(
	uint8_t const* pixels, int w, int h,
	std::array<std::vector<uint8_t>, 6>& faces_out,
	int& faceSizeOut
) {
	if (h != 6 * w) return false;
	int s = w;

	for (auto& f : faces_out) f.assign(size_t(s) * size_t(s) * 4, 0);
	faceSizeOut = s;

	for (int face = 0; face < 6; ++face) {
		int y0 = face * s;
		auto& dst = faces_out[face];
		for (int y = 0; y < s; ++y) {
			std::memcpy(
				dst.data() + size_t(y) * size_t(s) * 4,
				pixels + 4 * ((y0 + y) * w),
				size_t(s) * 4
			);
		}
	}
	return true;
}

[[maybe_unused]]  static bool split_cube_faces_vertical_strip_rgba8(
	uint8_t const* pixels, int w, int h,
	std::array<std::vector<uint8_t>, 6>& faces_out,
	int& faceSizeOut
) {
	if (h != 6 * w) return false;

	int s = w;
	faceSizeOut = s;

	for (auto& f : faces_out) {
		f.resize(size_t(s) * size_t(s) * 4);
	}

	//auto faceBytes = size_t(face_size) * size_t(face_size) * 4;
	//(void)faceBytes;

	for (int face = 0; face < 6; ++face) {
		int y0 = face * s;
		uint8_t* dst = faces_out[face].data();

		for (int y = 0; y < s; ++y) {
			uint8_t const* srcRow = pixels + 4 * ((y0 + y) * w);
			std::memcpy(dst + size_t(y) * size_t(s) * 4, srcRow, size_t(s) * 4);
		}
	}

	return true;
}
static Helpers::AllocatedImage create_cubemap_image_mips(
	RTG& rtg,
	VkExtent2D extent,
	VkFormat format,
	uint32_t mipLevels,
	VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties,
	Helpers::MapFlag map
) {
	Helpers::AllocatedImage image;
	image.extent = extent;
	image.format = format;

	VkImageCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent{
			.width = extent.width,
			.height = extent.height,
			.depth = 1
		},
		.mipLevels = mipLevels, // <-- HERE
		.arrayLayers = 6,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	//VK(vkCreateImage(rtg.device, &create_info, nullptr, &image.handle));
	VkResult res = vkCreateImage(rtg.device, &create_info, nullptr, &image.handle);
	if (res != VK_SUCCESS || image.handle == VK_NULL_HANDLE) {
		std::cout << "FAILED to create cubemap image\n";
		abort();
	}

	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(rtg.device, image.handle, &req);

	image.allocation = rtg.helpers.allocate(req, properties, map);
	VK(vkBindImageMemory(rtg.device, image.handle, image.allocation.handle, image.allocation.offset));

	return image;
}

static Helpers::AllocatedImage create_cubemap_image(
	RTG& rtg,
	VkExtent2D extent,
	VkFormat format,
	VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties,
	Helpers::MapFlag map
) {
	Helpers::AllocatedImage image;
	image.extent = extent;
	image.format = format;

	VkImageCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent{
			.width = extent.width,
			.height = extent.height,
			.depth = 1
		},
		.mipLevels = 1,
		.arrayLayers = 6, // 6 faces
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	VK(vkCreateImage(rtg.device, &create_info, nullptr, &image.handle));

	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(rtg.device, image.handle, &req);

	// reuse Helpers allocator:
	image.allocation = rtg.helpers.allocate(req, properties, map);

	VK(vkBindImageMemory(rtg.device, image.handle, image.allocation.handle, image.allocation.offset));

	return image;
}

static inline void rgbe_to_float4(uint8_t R, uint8_t G, uint8_t B, uint8_t E, float out[4]) {
	if (E == 0) { out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f; return; }

	// Standard Ward RGBE decode:
	// scale = 2^(E - 128) / 256
	float scale = std::ldexp(1.0f, int(E) - 128) / 256.0f;

	out[0] = (float(R) + 0.5f) * scale;
	out[1] = (float(G) + 0.5f) * scale;
	out[2] = (float(B) + 0.5f) * scale;
	out[3] = 1.0f;
}

 
static mat4 mat4_inverse(mat4 const& m);
[[maybe_unused]] static uint32_t load_png_texture_srgb_into(
	RTG& rtg,
	std::vector<Helpers::AllocatedImage>& textures,
	std::unordered_map<std::string, uint32_t>& cache,
	std::string const& tex_path
) {
	// Use UINT32_MAX as failure sentinel (0 is a valid texture index)
	constexpr uint32_t FAIL = UINT32_MAX;

	auto it = cache.find(tex_path);
	if (it != cache.end()) return it->second;

	int w = 0, h = 0, n = 0;
	stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
	if (!pixels) {
		std::cout << "[A2] failed to load env texture: " << tex_path << "\n";
		return FAIL;
	}

	textures.emplace_back(rtg.helpers.create_image(
		VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	));

	rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, textures.back());
	stbi_image_free(pixels);

	uint32_t idx = uint32_t(textures.size() - 1);
	cache[tex_path] = idx;
	return idx;
}
 
static void collect_scene_cameras(S72 const& scene, std::vector<S72::Node const*>& out);

// --- A2: find an active ENVIRONMENT (file-scope helpers) ---

static S72::Environment const* find_env_in_subtree(S72::Node const* node) {
	if (!node) return nullptr;

	if (node->environment) return node->environment;

	for (auto const* child : node->children) {
		if (auto* env = find_env_in_subtree(child)) return env;
	}
	return nullptr;
}

static S72::Environment const* find_active_environment(S72 const& scene, S72::Node const* camera_node) {
	// 1) if the camera node has an environment, use it
	if (camera_node && camera_node->environment) return camera_node->environment;

	// 2) otherwise: first environment found under any scene root
	for (auto const* root : scene.scene.roots) {
		if (auto* env = find_env_in_subtree(root)) return env;
	}

	// 3) none present
	return nullptr;
}

[[maybe_unused]] static bool split_cube_faces_vertical_strip_rgbe(
	uint8_t const* pixels, int w, int h,
	std::array<std::vector<float>, 6>& faces_out,
	int& faceSizeOut
) {
	if (h != 6 * w) return false;

	int s = w;
	for (auto& f : faces_out) f.assign(size_t(s) * size_t(s) * 4, 0.0f);
	faceSizeOut = s;

	for (int face = 0; face < 6; ++face) {
		auto& dst = faces_out[face];
		int x0 = 0;
		int y0 = face * s;

		for (int y = 0; y < s; ++y) {
			for (int x = 0; x < s; ++x) {
				uint8_t const* p = pixels + 4 * ((y0 + y) * w + (x0 + x));
				float rgba[4];
				rgbe_to_float4(p[0], p[1], p[2], p[3], rgba);

				size_t o = 4 * (size_t(y) * size_t(s) + size_t(x));
				dst[o + 0] = rgba[0];
				dst[o + 1] = rgba[1];
				dst[o + 2] = rgba[2];
				dst[o + 3] = rgba[3];
			}
		}
	}

	return true;
}

 

 


Tutorial::Tutorial(RTG& rtg_, std::string const& scene_file_, RTG::Configuration::CullingMode culling_mode_) : rtg(rtg_), scene_file(scene_file_) {

	//if (ft_log_enabled) ft_logger.stop();

	culling_mode = culling_mode_;
	enable_culling = (culling_mode_ == RTG::Configuration::CullingMode::Frustum);

	uint32_t rough_default_idx = 0;
	uint32_t metal_default_idx = 0;

	
	auto write_pbr_env_set3 = [&](VkDescriptorSet dst_set) {
		//if (!has_env_lambertian || !has_env_ggx || !has_brdf_lut) {
			//return; // skip if anything missing
		//}
		// fallback to dummy if missing
		 
		std::cout << "WRITE PBR ENV CALLED\n";

		VkImageView lamView =
			(env_lambertian_cubemap_view != VK_NULL_HANDLE)
			? env_lambertian_cubemap_view
			: env_cubemap_view;

		VkImageView ggxView =
			(env_ggx_cubemap_view != VK_NULL_HANDLE)
			? env_ggx_cubemap_view
			: env_cubemap_view;

		
		VkImageView brdfView =
			(brdf_lut_view != VK_NULL_HANDLE) ? brdf_lut_view : dummy_brdf_lut_view; // (keep it, no fallback for now)
		/*VkDescriptorImageInfo lam_info{
			.sampler = env_sampler,
			.imageView = env_lambertian_cubemap_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
		VkDescriptorImageInfo ggx_info{
			.sampler = env_sampler,
			.imageView = env_ggx_cubemap_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
		VkDescriptorImageInfo brdf_info{
			.sampler = env_sampler,
			.imageView = brdf_lut_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};*/

VkDescriptorImageInfo lam_info{
	.sampler = env_sampler,
	.imageView = lamView,
	.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
};
VkDescriptorImageInfo ggx_info{
	.sampler = env_sampler,
	.imageView = ggxView,
	.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
};
VkDescriptorImageInfo brdf_info{
	.sampler = env_sampler,
	.imageView = brdfView,
	.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
};

		std::array<VkWriteDescriptorSet, 3> writes{
			VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = dst_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &lam_info,
			},
			VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = dst_set,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &ggx_info,
			},
			VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = dst_set,
				.dstBinding = 2,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &brdf_info,
			},
		};

		std::cout
			<< "lam: " << env_lambertian_cubemap_view << "\n"
			<< "ggx: " << env_ggx_cubemap_view << "\n"
			<< "brdf: " << brdf_lut_view << "\n"
			<< "sampler: " << env_sampler << "\n";
		vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
		};



	use_s72_scene = !scene_file_.empty();
	if (use_s72_scene) {
		scene = S72::load(scene_file_);

		//light helper
		loaded_lights.clear(); //clears old runtime light data

		mat4 identity = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		}; 

		for (auto const* root : scene.scene.roots) { //walk root nodes
			collect_loaded_lights_from_node(root, identity, loaded_lights); //ollects all node-instanced lights
			//into loaded_lights
		}

		std::cout << "loaded lights: " << loaded_lights.size() << std::endl; //prints count so
		//we can verify A3-load is actually working

		//shadow casting spot light build
		shadow_spot_lights.clear(); //clear old runtime shadow data
		for (auto& light : loaded_lights) { //go through all loaded lights
			if (light.type == LoadedLight::Type::Spot && light.shadow > 0.0f) {//only if spot light with shadow value > 0
				shadow_spot_lights.push_back(&light); //add specific light corresponding to criteration to list
			}
		}
		std::cout << "shadow spot lights: " << shadow_spot_lights.size() << std::endl; //good old log of count so you can verify it works

		
		auto make_dummy_brdf_lut = [&]() {
			if (dummy_brdf_lut_view != VK_NULL_HANDLE) return;

			uint8_t pixel[4] = { 255, 255, 255, 255 };

			dummy_brdf_lut = rtg.helpers.create_image(
				VkExtent2D{ 1, 1 },
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			rtg.helpers.transfer_to_image(pixel, sizeof(pixel), dummy_brdf_lut);

			VkImageViewCreateInfo view_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = dummy_brdf_lut.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_R8G8B8A8_UNORM,
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VK(vkCreateImageView(rtg.device, &view_info, nullptr, &dummy_brdf_lut_view));
			};
		// --- local fallback env cubemap  ---
		auto make_dummy_env_cubemap = [&]() {
			// If already made, don't remake:
			if (env_cubemap_view != VK_NULL_HANDLE) return;

			// 1x1 white per face (RGBA8)
			std::array<std::vector<uint8_t>, 6> faces{};
			for (int f = 0; f < 6; ++f) {
				faces[f].assign(4, 0);
			}

			if (env_cubemap.handle != VK_NULL_HANDLE) {
				rtg.helpers.destroy_image(std::move(env_cubemap));
				env_cubemap = {};
			}
			env_cubemap = create_cubemap_image(
				rtg,
				VkExtent2D{ 1, 1 },
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			VkImageViewCreateInfo view_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = env_cubemap.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_CUBE,
				.format = env_cubemap.format,
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 6,
				},
			};
			VK(vkCreateImageView(rtg.device, &view_info, nullptr, &env_cubemap_view));


			upload_cubemap_faces_rgba8(rtg, rtg.helpers.transfer_command_pool, env_cubemap, 1, faces);

			has_env_texture = false; // dummy counts as "no real env"
			};



		//load scene 
		std::cout << "[A1-load] scene: " << scene_file << "\n";
		std::cout << "  nodes:      " << scene.nodes.size() << "\n";
		std::cout << "  meshes:     " << scene.meshes.size() << "\n";
		std::cout << "  cameras:    " << scene.cameras.size() << "\n";
		std::cout << "  materials:  " << scene.materials.size() << "\n";
		std::cout << "  textures:   " << scene.textures.size() << "\n";
		std::cout << "  datafiles:  " << scene.data_files.size() << "\n";

		collect_scene_cameras(scene, scene_camera_nodes);
		std::cout << "[A1-show] scene cameras found: " << scene_camera_nodes.size() << "\n";

		// pick defaults:
		if (!scene_camera_nodes.empty()) active_scene_camera = 0;

		// give debug camera a sensible starting pose:
		debug_camera = free_camera;

		// --- environment selection   ---
		S72::Node const* active_camera_node =
			(!scene_camera_nodes.empty() ? scene_camera_nodes[active_scene_camera] : nullptr);

		S72::Environment const* active_env = find_active_environment(scene, active_camera_node);

		if (active_env) {
			std::cout << "[A2] Active environment: " << active_env->name
				<< " radiance src=" << active_env->radiance->src
				<< " path=" << active_env->radiance->path
				<< "\n";
		}
		// --- A2: we found an environment ---
		has_env_texture = (active_env && active_env->radiance);
		env_texture = UINT32_MAX;

		{ // A2 env sampler (cube-friendly)
			VkSamplerCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.minLod = 0.0f,
				.maxLod = 9.0,
				.unnormalizedCoordinates = VK_FALSE,
			};
			VK(vkCreateSampler(rtg.device, &create_info, nullptr, &env_sampler));
			make_dummy_env_cubemap();
			make_dummy_brdf_lut();
		}
		 

		

		if (active_env && active_env->radiance) {
 
			// env load:
			int w = 0, h = 0, n = 0;
			stbi_uc* pixels = stbi_load(active_env->radiance->path.c_str(), &w, &h, &n, 4);

			std::cout << "env: " << active_env->radiance->path << " -> " << w << "x" << h << " n=" << n << "\n";
			// lambertian load (for A2-diffuse):
			std::filesystem::path env_path(active_env->radiance->path);

			// If your lambertian file is named like: env_rgbe.lambertian.png
			std::filesystem::path lam_path =
				env_path.parent_path() / (env_path.stem().string() + ".lambertian.png");

			// Debug: where are we looking / does it exist:
			std::cout << "cwd: " << std::filesystem::current_path().string() << "\n";
			std::cout << "lam path: " << lam_path.string() << "\n";
			std::cout << "lam exists? " << std::filesystem::exists(lam_path) << "\n";

			int lw = 0, lh = 0, ln = 0;
			stbi_uc* lam_pixels = stbi_load(lam_path.string().c_str(), &lw, &lh, &ln, 4);

			std::cout << "lam: " << lam_path.string() << " -> " << lw << "x" << lh << " n=" << ln << "\n";

			has_env_lambertian = false;

			if (!lam_pixels) {
				std::cout << "[A2] failed to load lambertian image: " << lam_path << "\n";
			}
			else {
				std::array<std::vector<float>, 6> lam_faces_f;
				int lam_faceSize = 0;

				if (!split_cube_faces_vertical_strip_rgbe(lam_pixels, lw, lh, lam_faces_f, lam_faceSize)) {
					std::cout << "[A2] lambertian image not a 6x vertical strip: " << lw << "x" << lh << "\n";
				}
				else {
					// create float cubemap image for irradiance:
					env_lambertian_cubemap = create_cubemap_image(
						rtg,
						VkExtent2D{ uint32_t(lam_faceSize), uint32_t(lam_faceSize) },
						VK_FORMAT_R32G32B32A32_SFLOAT,
						VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						Helpers::Unmapped
					);

					VkImageViewCreateInfo lam_view_info{
						.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
						.image = env_lambertian_cubemap.handle,
						.viewType = VK_IMAGE_VIEW_TYPE_CUBE,
						.format = env_lambertian_cubemap.format,
						.subresourceRange{
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.baseMipLevel = 0,
							.levelCount = 1,
							.baseArrayLayer = 0,
							.layerCount = 6,
						},
					};
					VK(vkCreateImageView(rtg.device, &lam_view_info, nullptr, &env_lambertian_cubemap_view));

					// upload float faces:
					upload_cubemap_faces_float4(rtg, env_lambertian_cubemap, lam_faceSize, lam_faces_f);

					has_env_lambertian = true;
				}

				stbi_image_free(lam_pixels);
			}

			if (!pixels) {
				std::cout << "[A2] failed to load env image: " << active_env->radiance->path << "\n";
				has_env_texture = false;
			}
			else {
				// RGBE -> float faces
				std::array<std::vector<float>, 6> faces_f;
				int faceSize = 0;

				if (!split_cube_faces_vertical_strip_rgbe(pixels, w, h, faces_f, faceSize)) {
					std::cout << "[A2] env image not a 6x vertical strip: " << w << "x" << h << "\n";
					has_env_texture = false;
					stbi_image_free(pixels);
				}
				else {
					stbi_image_free(pixels);

					if (env_cubemap.handle != VK_NULL_HANDLE) {
						rtg.helpers.destroy_image(std::move(env_cubemap));
						env_cubemap = {};
					}
					env_cubemap = create_cubemap_image(
						rtg,
						VkExtent2D{ uint32_t(faceSize), uint32_t(faceSize) },
						VK_FORMAT_R32G32B32A32_SFLOAT, // 
						VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						Helpers::Unmapped
					);

					VkImageViewCreateInfo view_info{
						.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
						.image = env_cubemap.handle,
						.viewType = VK_IMAGE_VIEW_TYPE_CUBE,
						.format = env_cubemap.format,
						.subresourceRange{
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.baseMipLevel = 0,
							.levelCount = 1,
							.baseArrayLayer = 0,
							.layerCount = 6,
						},
					};
					VK(vkCreateImageView(rtg.device, &view_info, nullptr, &env_cubemap_view));

					// upload float faces (staging now uses sizeof(float)*4 per pixel)
					upload_cubemap_faces_float4(rtg, env_cubemap, faceSize, faces_f);

					has_env_texture = true;
				}
			}
			// -------------------- GGX prefiltered specular cubemap (mip chain) --------------------
			has_env_ggx = false;

			std::filesystem::path ggx_base =
				env_path.parent_path() / (env_path.stem().string() + ".ggx"); // ".../ox_bridge_morning.ggx"
			std::cout << "ggx base: " << ggx_base.string() << "\n";

			// will be filled from mip0:
			uint32_t ggxBaseSize = 0;
			uint32_t ggxMipLevels = 0;

			// ---- load mip0 to discover base size + mip count ----
			std::filesystem::path mip0_path = ggx_base;
			mip0_path += "_0.png";

			int w0 = 0, h0 = 0, n0 = 0;
			stbi_uc* mip0_pixels = stbi_load(mip0_path.string().c_str(), &w0, &h0, &n0, 4);
			if (!mip0_pixels) {
				std::cout << "[A2] failed to load ggx mip0: " << mip0_path << "\n";
				has_env_ggx = false;
			}
			else {

			std::array<std::vector<float>, 6> mip0_faces;
			int faceSize0 = 0;
			if(!split_cube_faces_vertical_strip_rgbe(mip0_pixels, w0, h0, mip0_faces, faceSize0)) {
				std::cout << "...";
				stbi_image_free(mip0_pixels);
				has_env_ggx = false;
			}
			else {
				stbi_image_free(mip0_pixels);

				ggxBaseSize = uint32_t(faceSize0);

				// mipLevels = floor(log2(baseSize)) + 1
				ggxMipLevels = 1;
				for (uint32_t t = ggxBaseSize; t > 1; t >>= 1) ggxMipLevels++;

				std::cout << "[A2] ggxBaseSize=" << ggxBaseSize << " ggxMipLevels=" << ggxMipLevels << "\n";

				// ---- create cubemap with mip chain ----
				env_ggx_cubemap = create_cubemap_image_mips(
					rtg,
					VkExtent2D{ ggxBaseSize, ggxBaseSize },
					VK_FORMAT_R32G32B32A32_SFLOAT,
					ggxMipLevels,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					Helpers::Unmapped
				);

				// view includes all mips:
				VkImageViewCreateInfo ggx_view_info{
					.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
					.image = env_ggx_cubemap.handle,
					.viewType = VK_IMAGE_VIEW_TYPE_CUBE,
					.format = env_ggx_cubemap.format,
					.subresourceRange{
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = ggxMipLevels,
						.baseArrayLayer = 0,
						.layerCount = 6,
					},
				};
				VK(vkCreateImageView(rtg.device, &ggx_view_info, nullptr, &env_ggx_cubemap_view));

				// ---- upload mip0 (we already have it) ----
				upload_cubemap_faces_float4_mip(rtg, env_ggx_cubemap, faceSize0, 0, mip0_faces);
				has_env_ggx = true;


				// ---- load + upload mip 1..N-1 ----
				for (uint32_t mip = 1; mip < ggxMipLevels; ++mip) {
					uint32_t mipSize = ggxBaseSize >> mip;
					if (mipSize < 1) mipSize = 1;

					std::filesystem::path mip_path = ggx_base;
					mip_path += "_" + std::to_string(mip) + ".png"; // ".../something.ggx_1.png"

					int mw = 0, mh = 0, mn = 0;
					stbi_uc* ggx_pixels = stbi_load(mip_path.string().c_str(), &mw, &mh, &mn, 4);
					std::cout << "ggx mip " << mip << ": " << mip_path.string() << " -> " << mw << "x" << mh << " n=" << mn << "\n";

					if (!ggx_pixels) {
						std::cout << "[A2] failed to load ggx mip: " << mip_path << "\n";
						has_env_ggx = false;
						break;
					}

					std::array<std::vector<float>, 6> ggx_faces_f;
					int ggx_faceSize = 0;

					if (!split_cube_faces_vertical_strip_rgbe(ggx_pixels, mw, mh, ggx_faces_f, ggx_faceSize)) {
						std::cout << "[A2] ggx mip not a 6x vertical strip: " << mw << "x" << mh << "\n";
						stbi_image_free(ggx_pixels);
						has_env_ggx = false;
						break;
					}
					stbi_image_free(ggx_pixels);

					if (uint32_t(ggx_faceSize) != mipSize) {
						std::cout << "[A2] ggx mip size mismatch. expected " << mipSize << " got " << ggx_faceSize << "\n";
						has_env_ggx = false;
						break;
					}

					upload_cubemap_faces_float4_mip(rtg, env_ggx_cubemap, ggx_faceSize, mip, ggx_faces_f);
				}
			}
			
			}

			// -------------------- BRDF LUT (load ONCE, not per mip) --------------------
			has_brdf_lut = false;

			std::filesystem::path brdf_path = env_path.parent_path() / "brdf_lut.png";
			std::cout << "brdf lut path: " << brdf_path.string() << "\n";

			// 1. Force 4 channels (RGBA) even if the file is just RG
			int bw = 0, bh = 0, bn = 0;
			stbi_uc* brdf_pixels = stbi_load(brdf_path.string().c_str(), &bw, &bh, &bn, 4);

			if (!brdf_pixels) {
				std::cout << "[A2] failed to load brdf lut: " << brdf_path << "\n";
			}
			else {
				// 2. Use the standard RGBA format (4 bytes per pixel)
				brdf_lut = rtg.helpers.create_image(
					VkExtent2D{ uint32_t(bw), uint32_t(bh) },
					VK_FORMAT_R8G8B8A8_UNORM,
					VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					Helpers::Unmapped
				);

				// 3. Size is now definitely width * height * 4
				rtg.helpers.transfer_to_image(brdf_pixels, size_t(bw) * size_t(bh) * 4, brdf_lut);
				stbi_image_free(brdf_pixels);

				// 4. Create the view using the new RGBA format
				VkImageViewCreateInfo view_info{
					.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
					.image = brdf_lut.handle,
					.viewType = VK_IMAGE_VIEW_TYPE_2D,
					.format = VK_FORMAT_R8G8B8A8_UNORM, // Match the image!
					.subresourceRange{
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					},
				};
				VK(vkCreateImageView(rtg.device, &view_info, nullptr, &brdf_lut_view));

				has_brdf_lut = true;
			}

		

		

		
		}

		if (!has_env_texture) {
			make_dummy_env_cubemap();
			make_dummy_brdf_lut();
		}
	}else{
		// no scene; keep fallback mode
		scene_camera_nodes.clear();
		active_scene_camera = 0;
		debug_camera = free_camera;
	}

	// --- enforce --camera <name> requirement :
	if (rtg.configuration.camera_name.has_value() && !rtg.configuration.camera_name->empty()) {
		const std::string& requested = *rtg.configuration.camera_name;
		 
		if (!use_s72_scene) {
			throw std::runtime_error("--camera was provided, but no scene file was loaded.");
		}

		if (scene_camera_nodes.empty()) {
			throw std::runtime_error("--camera was provided, but the scene contains no cameras.");
		}

		camera_mode = CameraMode::Scene;

		bool found = false;
		for (uint32_t i = 0; i < uint32_t(scene_camera_nodes.size()); ++i) {
			S72::Node const& n = *scene_camera_nodes[i];


			std::string node_name;
			std::string camera_name;

			// If Node has .name:
			if constexpr (requires { n.name; }) {
				node_name = n.name;
			}

			// If camera exists and has .name:
			if (n.camera) {
				if constexpr (requires { n.camera->name; }) {
					camera_name = n.camera->name;
				}
			}

			if (node_name == rtg.configuration.camera_name || camera_name == rtg.configuration.camera_name) {
				active_scene_camera = i;
				found = true;
				break;
			}
		}

		if (!found) {
			// optional: print available camera names to help debugging
			std::string available;
			for (S72::Node const* p : scene_camera_nodes) {
				if (!p) continue;
				if constexpr (requires { p->name; }) {
					available += "  - " + p->name + "\n";
				}
				else {
					available += "  - (unnamed)\n";
				}
			}

			throw std::runtime_error(
				"No camera named '" + requested + "' in scene.\n"
				"Available cameras:\n" + available
			);
		}

		
	}



	

	 
	auto read_vec4_f32 = [&](std::vector<uint8_t> const& bytes, size_t byte_offset) -> std::array<float, 4> {
		if (byte_offset + 16 > bytes.size()) return { 0.0f, 0.0f, 0.0f, 1.0f };
		float const* f = reinterpret_cast<float const*>(bytes.data() + byte_offset);
		return { f[0], f[1], f[2], f[3] };
		};
		
	
	

	 
	// select a depth format:
	depth_format = rtg.helpers.find_image_format(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32 },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);

	shadow_maps.clear();
	shadow_map_views.clear();
	shadow_framebuffers.clear();

	shadow_maps.resize(shadow_spot_lights.size());
	shadow_map_views.resize(shadow_spot_lights.size(), VK_NULL_HANDLE);
	shadow_framebuffers.resize(shadow_spot_lights.size(), VK_NULL_HANDLE);

	for (size_t li = 0; li < shadow_spot_lights.size(); ++li) {

		uint32_t shadowMapSize = uint32_t(shadow_spot_lights[li]->shadow);

		shadow_maps[li] = rtg.helpers.create_image(
			VkExtent2D{ shadowMapSize, shadowMapSize },
			depth_format,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		VkImageViewCreateInfo shadow_view_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = shadow_maps[li].handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VK(vkCreateImageView(rtg.device, &shadow_view_info, nullptr, &shadow_map_views[li]));
	}

	VkSamplerCreateInfo shadow_sampler_info{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		.unnormalizedCoordinates = VK_FALSE,
	};
	VK(vkCreateSampler(rtg.device, &shadow_sampler_info, nullptr, &shadow_sampler));

	{//shadow render pass
		VkAttachmentDescription depth_attachment{
			.format = depth_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		};

		VkAttachmentReference depth_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 0,
			.pDepthStencilAttachment = &depth_ref,
		};

		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &depth_attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
		};

		VK(vkCreateRenderPass(rtg.device, &create_info, nullptr, &shadow_render_pass));
	}
	for (size_t li = 0; li < shadow_map_views.size(); ++li) {
		VkFramebufferCreateInfo fb_info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = shadow_render_pass,
			.attachmentCount = 1,
			.pAttachments = &shadow_map_views[li],
			.width = shadow_maps[li].extent.width,
			.height = shadow_maps[li].extent.height,
			.layers = 1,
		};
		VK(vkCreateFramebuffer(rtg.device, &fb_info, nullptr, &shadow_framebuffers[li]));
	}

	 
	{ //create render pass
		//attachments:
		std::array< VkAttachmentDescription, 2 > attachments{
			VkAttachmentDescription{ //0 - color attachment:
				.format = rtg.surface_format.format, //DEFINE FORMAT
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, //LOADOP LOAD THE DATA
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE, //how to write data back after rendering
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, //LAYOUT IMAGE TRANSITIONED TO BEFORE THE LOAD
				.finalLayout = rtg.present_layout, //layout image is transitioned to after the store
			     
			},
			VkAttachmentDescription{ //1 - depth attachment:
				.format = depth_format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				},
		};

		// subpass ( parts of the rendering that can proceed (potentially) in parallel)
		VkAttachmentReference color_attachment_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_attachment_ref{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref,
			.pDepthStencilAttachment = &depth_attachment_ref,
		};

		//dependencies
		//this defers the image load actions for the attachments:
		std::array< VkSubpassDependency, 2> dependencies{
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				},
				VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				}
		};


		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = uint32_t(dependencies.size()),
			.pDependencies = dependencies.data(),
		};

		VK(vkCreateRenderPass(rtg.device, &create_info, nullptr, &render_pass));
	}

	
	{//create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK(vkCreateCommandPool(rtg.device, &create_info, nullptr, &command_pool));
	}

	//calling create function fron tutorial.hppp

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);
	objects_pipeline.create(rtg, render_pass, 0);
	pbr_pipeline.create(rtg, render_pass, 0);
	mirror_pipeline.create(
		rtg,
		render_pass,
		0,
		background_pipeline.set_layout,          // env layout
		objects_pipeline.set1_Transforms         // transforms layout
	);
	shadow_pipeline.create(rtg, shadow_render_pass, 0);

	 
	 

	{ //create descriptor pool:
		uint32_t per_workspace = uint32_t(rtg.workspaces.size()); //for easier-to-read counting

		std::array<VkDescriptorPoolSize, 3> pool_sizes{
	VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 * per_workspace},
	VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20 * per_workspace}, // merge both
	VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 20 * per_workspace},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = 30 * per_workspace, // ← move it HERE
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &descriptor_pool));
	}


	workspaces.resize(rtg.workspaces.size());
	for (Workspace& workspace : workspaces) {
		

		

		{//allocate command buffer:
			VkCommandBufferAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &workspace.command_buffer));
		}

		 

		workspace.Camera_src = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible
			//memory, coherent (no special sync needed)
			Helpers::Mapped //get a pointer to the memory

		);
		workspace.Camera = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as a uniform
			//buffer, also going to have GPU copy into this memory
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
			Helpers::Unmapped //don't get a pointer to the memory
		);

		 

		 

		 

		 
		//descriptor set:
		{ //allocate descriptor set for Camera descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lines_pipeline.set0_Camera,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Camera_descriptors));
		}

		workspace.World_src = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);
		workspace.World = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{ //allocate descriptor set for World descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set0_World,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.World_descriptors));
			//NOTE: will actually fill in this descriptor set just a bit lower
		}

		{ //allocate descriptor set for Transforms descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set1_Transforms,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Transforms_descriptors));
			//  will fill in this descriptor set in render when buffers are [re-allocated]
		}

		{ // allocate descriptor set for PBR env (set3: lambertian + ggx + brdf lut)
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &pbr_pipeline.set3_EnvPBR, // this layout has 3 bindings in your PBRPipeline
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.PBR_Env_descriptors));
		}

		

		workspace.Lights = rtg.helpers.create_buffer(
			sizeof(GPULight) * std::max<size_t>(size_t(1), loaded_lights.size()),
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);

		std::cout << "ALLOC using set4_Lights: " << pbr_pipeline.set4_Lights << std::endl;
		{ // allocate descriptor set for Lights (set 4)
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &pbr_pipeline.set4_Lights,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Lights_descriptors));
		}

		{ // allocate descriptor set for Shadow (set 5)
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &pbr_pipeline.set5_Shadow,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Shadow_descriptors));
		}

		workspace.Shadow_descriptors_per_light.resize(shadow_spot_lights.size(), VK_NULL_HANDLE);

		for (size_t li = 0; li < shadow_spot_lights.size(); ++li) {
			VkDescriptorSetAllocateInfo shadow_alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &pbr_pipeline.set5_Shadow, // use the same shadow set layout you already use
			};

			VK(vkAllocateDescriptorSets(
				rtg.device,
				&shadow_alloc_info,
				&workspace.Shadow_descriptors_per_light[li]
			));
		}

		 
		 

		// descriptor write:
		{
			VkDescriptorBufferInfo Camera_info{
				.buffer = workspace.Camera.handle,
				.offset = 0,
				.range = workspace.Camera.size,
			};

			VkDescriptorBufferInfo World_info{
				.buffer = workspace.World.handle,
				.offset = 0,
				.range = workspace.World.size,
			};

			 // write Lights descriptor (set 4)
				VkDescriptorBufferInfo lights_info{
					.buffer = workspace.Lights.handle,
					.offset = 0,
					.range = VK_WHOLE_SIZE
				};

				constexpr uint32_t MAX_SHADOW_SPOT_LIGHTS = 16;
				std::array<VkDescriptorImageInfo, MAX_SHADOW_SPOT_LIGHTS> shadow_infos{};
				VkImageView fallback_view = shadow_map_views.empty() ? VK_NULL_HANDLE : shadow_map_views[0];
				for (uint32_t si = 0; si < MAX_SHADOW_SPOT_LIGHTS; ++si) {
					VkImageView v = fallback_view;
					if (si < shadow_map_views.size()) v = shadow_map_views[si];
					shadow_infos[si] = VkDescriptorImageInfo{
						.sampler = shadow_sampler,
						.imageView = v,
						.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
					};
				}

				VkWriteDescriptorSet shadow_write{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Shadow_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = MAX_SHADOW_SPOT_LIGHTS,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = shadow_infos.data(),
				};

				for (size_t li = 0; li < shadow_spot_lights.size(); ++li) {
					VkDescriptorImageInfo shadow_info_per_light{
						.sampler = shadow_sampler,
						.imageView = shadow_map_views[li],
						.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
					};

					VkWriteDescriptorSet shadow_write_per_light{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = workspace.Shadow_descriptors_per_light[li],
						.dstBinding = 0,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &shadow_info_per_light,
					};

					vkUpdateDescriptorSets(rtg.device, 1, &shadow_write_per_light, 0, nullptr);
				}
				 
				{
					
					
					write_pbr_env_set3(workspace.PBR_Env_descriptors);

				}

			 


			// --- ALWAYS write camera + world ---
			std::array<VkWriteDescriptorSet, 2 > base_writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Camera_descriptors,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &Camera_info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.World_descriptors,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &World_info,
				}
			};

			VkWriteDescriptorSet write{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Lights_descriptors,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &lights_info
			};

			

		 


			vkUpdateDescriptorSets(rtg.device, uint32_t(base_writes.size()), base_writes.data(), 0, nullptr);
			vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);
			vkUpdateDescriptorSets(rtg.device, 1, &shadow_write, 0, nullptr);
			 
			{
				// Create a temporary CPU array that matches the GPU layout
				std::vector<GPULight> gpu_lights;

				// Avoid reallocations (performance, not correctness)
				gpu_lights.reserve(loaded_lights.size());

				// Convert each scene light into GPU format
				for (auto const& light : loaded_lights) {

					GPULight gpu{};

					// --- Encode light type ---
					// Shader doesn't understand enums, so we pack it into a float
					float type_value = 0.0f;
					if (light.type == LoadedLight::Type::Sun) type_value = 0.0f;
					else if (light.type == LoadedLight::Type::Sphere) type_value = 1.0f;
					else if (light.type == LoadedLight::Type::Spot) type_value = 2.0f;

					// --- Position ---
					// xyz = world position
					// w   = type (Sun / Sphere / Spot)
					gpu.position = {
						light.world_position.x,
						light.world_position.y,
						light.world_position.z,
						type_value
					};

					// --- Direction ---
					// xyz = direction (for spot/sun)
					// w   = shadow flag (currently unused, but preserved)
					gpu.direction = {
	light.world_direction.x,
	light.world_direction.y,
	light.world_direction.z,
	light.blend
					};

					// --- Tint (color) ---
						// xyz = light color
						// w   = shadow map size (used by shader to identify shadow-casting spots)
					gpu.tint = {
						light.tint.x,
						light.tint.y,
						light.tint.z,
						light.shadow
					};

					// --- Parameters ---
					// Sun:    x = angle, y = strength
					// Sphere: x = radius, y = power, z = limit
					// Spot:   x = radius, y = power, z = limit, w = fov
					if (light.type == LoadedLight::Type::Sun) {
						gpu.params = { light.angle, light.strength, 0.0f, 0.0f };
					}
					else {
						gpu.params = { light.radius, light.power, light.limit, light.fov };
					}

					// Store this light into the array
					gpu_lights.emplace_back(gpu);
				}

				// --- Upload to GPU buffer ---
				// Only copy if we actually have lights
				if (!gpu_lights.empty()) {

					// workspace.Lights is already mapped CPU-visible memory
					// So memcpy writes directly into the buffer used by the GPU
					std::memcpy(
						workspace.Lights.allocation.data(),
						gpu_lights.data(),
						sizeof(GPULight) * gpu_lights.size()
					);
				}
			}

			// --- ONLY write env if valid ---
			if (has_env_lambertian && has_env_ggx && has_brdf_lut) {

				VkDescriptorImageInfo lam_info{
					.sampler = env_sampler,
					.imageView = env_lambertian_cubemap_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				VkDescriptorImageInfo ggx_info{
					.sampler = env_sampler,
					.imageView = env_ggx_cubemap_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				VkDescriptorImageInfo brdf_info{
					.sampler = env_sampler,
					.imageView = brdf_lut_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				std::array<VkWriteDescriptorSet, 3> env_writes{
					VkWriteDescriptorSet{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = workspace.PBR_Env_descriptors,
						.dstBinding = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &lam_info,
					},
					VkWriteDescriptorSet{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = workspace.PBR_Env_descriptors,
						.dstBinding = 1,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &ggx_info,
					},
					VkWriteDescriptorSet{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = workspace.PBR_Env_descriptors,
						.dstBinding = 2,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &brdf_info,
					}
				};

				vkUpdateDescriptorSets(rtg.device, uint32_t(env_writes.size()), env_writes.data(), 0, nullptr);
			}

			 
 

			 
		}


	}
	 

	{ //Pack S72 meshes into a single vertex buffer 

	//--- cache raw bytes for each DataFile so we only read each file once:
	std::unordered_map< S72::DataFile const*, std::vector<uint8_t> > data_cache;

	auto load_datafile = [&](S72::DataFile const& df) -> std::vector<uint8_t> const& {
		S72::DataFile const* key = &df;
		auto it = data_cache.find(key);
		if (it != data_cache.end()) return it->second;

		std::ifstream in(df.path, std::ios::binary);
		if (!in) {
			throw std::runtime_error("Failed to open data file: " + df.path);
		}

		in.seekg(0, std::ios::end);
		std::streamsize sz = in.tellg();
		in.seekg(0, std::ios::beg);

		std::vector<uint8_t> bytes;
		bytes.resize(size_t(sz));
		if (sz > 0) {
			in.read(reinterpret_cast<char*>(bytes.data()), sz);
		}

		auto [inserted_it, ok] = data_cache.emplace(key, std::move(bytes));
		return inserted_it->second;
		};

	auto find_attr = [&](S72::Mesh const& mesh, std::initializer_list<const char*> names) -> S72::Mesh::Attribute const* {
		for (auto n : names) {
			auto it = mesh.attributes.find(n);
			if (it != mesh.attributes.end()) return &it->second;
		}
		return nullptr;
		};

	auto read_vec3_f32 = [&](std::vector<uint8_t> const& bytes, size_t byte_offset) -> S72::vec3 {
		// assumes little-endian float32, which is what your class data will be
		if (byte_offset + 12 > bytes.size()) return S72::vec3{ 0,0,0 };
		float const* f = reinterpret_cast<float const*>(bytes.data() + byte_offset);
		return S72::vec3{ f[0], f[1], f[2] };
		};

	auto read_vec2_f32 = [&](std::vector<uint8_t> const& bytes, size_t byte_offset) -> std::pair<float, float> {
		if (byte_offset + 8 > bytes.size()) return { 0.0f, 0.0f };
		float const* f = reinterpret_cast<float const*>(bytes.data() + byte_offset);
		return { f[0], f[1] };
		};

	auto dot3 = [](S72::vec3 a, S72::vec3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
		};

	auto cross3 = [](S72::vec3 a, S72::vec3 b) {
		return S72::vec3{
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		};
		};

	auto add3 = [](S72::vec3 a, S72::vec3 b) {
		return S72::vec3{ a.x + b.x, a.y + b.y, a.z + b.z };
		};

	auto sub3 = [](S72::vec3 a, S72::vec3 b) {
		return S72::vec3{ a.x - b.x, a.y - b.y, a.z - b.z };
		};

	auto mul3s = [](S72::vec3 a, float s) {
		return S72::vec3{ a.x * s, a.y * s, a.z * s };
		};

	auto normalize3 = [&](S72::vec3 v) {
		float d2 = dot3(v, v);
		if (d2 <= 0.0f) return S72::vec3{ 1,0,0 };
		float inv = 1.0f / std::sqrt(d2);
		return mul3s(v, inv);
		};

	auto read_index = [&](std::vector<uint8_t> const& bytes, size_t byte_offset, VkIndexType type) -> uint32_t {
		if (type == VK_INDEX_TYPE_UINT16) {
			if (byte_offset + 2 > bytes.size()) return 0;
			uint16_t const* p = reinterpret_cast<uint16_t const*>(bytes.data() + byte_offset);
			return uint32_t(*p);
		}
		else if (type == VK_INDEX_TYPE_UINT32) {
			if (byte_offset + 4 > bytes.size()) return 0;
			uint32_t const* p = reinterpret_cast<uint32_t const*>(bytes.data() + byte_offset);
			return *p;
		}
		else {
			// unsupported index type
			return 0;
		}
		};

	//--- pack everything:
	std::vector< PosNorTexVertex > packed;
	packed.reserve(4096);

	s72_mesh_to_range.clear();


	//   scene.meshes is an unordered_map, so iteration order is arbitrary.
	// That's fine for now as long as we build instances using Mesh* later, not by index.
	for (auto const& kv : scene.meshes) {
		S72::Mesh const& mesh = kv.second;
		S72::Mesh const* mesh_ptr = &mesh;


		ObjectVertices range;
		range.first = uint32_t(packed.size());

		// We’ll support the common attribute names used in s72/glTF style exports.
		// If your exporter uses different keys, add them here.
		S72::Mesh::Attribute const* posA = find_attr(mesh, { "POSITION", "position", "pos" });
		S72::Mesh::Attribute const* norA = find_attr(mesh, { "NORMAL", "normal", "nor" });
		S72::Mesh::Attribute const* uvA = find_attr(mesh, { "TEXCOORD", "TEXCOORD_0", "texcoord", "uv", "UV" });
		S72::Mesh::Attribute const* tanA = find_attr(mesh, { "TANGENT", "tangent", "tan" });
		std::vector<uint8_t> const* tanBytes = nullptr;
		if (tanA) tanBytes = &load_datafile(tanA->src);

		if (!posA) {
			std::cout << "[A1-load] mesh '" << mesh.name << "' has no POSITION attribute; skipping.\n";
			range.count = 0;
			s72_mesh_to_range[mesh_ptr] = range;

			continue;
		}

		// Basic format sanity (you can relax later if needed):
		if (posA->format != VK_FORMAT_R32G32B32_SFLOAT) {
			std::cout << "[A1-load] mesh '" << mesh.name << "' POSITION format not vec3 f32; skipping.\n";
			range.count = 0;
			s72_mesh_to_range[mesh_ptr] = range;

			continue;
		}
		if (norA && norA->format != VK_FORMAT_R32G32B32_SFLOAT) {
			norA = nullptr; // ignore weird normals for now
		}
		if (uvA && uvA->format != VK_FORMAT_R32G32_SFLOAT) {
			uvA = nullptr; // ignore weird uvs for now
		}

		// Load the underlying data files:
		auto const& posBytes = load_datafile(posA->src);
		std::vector<uint8_t> const* norBytes = nullptr;
		std::vector<uint8_t> const* uvBytes = nullptr;

		if (norA) norBytes = &load_datafile(norA->src);
		if (uvA)  uvBytes = &load_datafile(uvA->src);

	 
// Tangent build (A2-normal)
//
// We compute tangents per *original vertex index* (the index into POSITION/NORMAL/UV arrays).
// Then when we expand indices into `packed`, we just attach the tangent for that vertex.
 

// how many vertices exist in the POSITION stream?
 
		uint32_t vertex_count = 0;
		{
			size_t available = 0;
			if (posBytes.size() > size_t(posA->offset)) {
				available = posBytes.size() - size_t(posA->offset);
			}
			if (posA->stride > 0) {
				vertex_count = uint32_t(available / size_t(posA->stride));
			}
		}

		// per-vertex accumulators:
		std::vector<S72::vec3> tan_sum(vertex_count, S72::vec3{ 0,0,0 });
		std::vector<S72::vec3> bitan_sum(vertex_count, S72::vec3{ 0,0,0 });

		// final tangent storage (xyz + w sign):
		struct Tan4 { float x, y, z, w; };
		std::vector<Tan4> tangents(vertex_count, Tan4{ 1,0,0,1 });

		// if we don't have UVs or normals, we can't do real tangents.
		// we will just leave the default (1,0,0, +1) and move on.
		bool can_build_tangents = (uvA && uvBytes && norA && norBytes && vertex_count > 0);

		// helper to read position/normal/uv for an original vertex index:
		auto read_pos = [&](uint32_t vi) -> S72::vec3 {
			size_t off = size_t(posA->offset) + size_t(vi) * size_t(posA->stride);
			return read_vec3_f32(posBytes, off);
			};
		auto read_nor = [&](uint32_t vi) -> S72::vec3 {
			if (!(norA && norBytes)) return S72::vec3{ 0,0,1 };
			size_t off = size_t(norA->offset) + size_t(vi) * size_t(norA->stride);
			return read_vec3_f32(*norBytes, off);
			};
		auto read_uv = [&](uint32_t vi) -> std::pair<float, float> {
			if (!(uvA && uvBytes)) return { 0.0f, 0.0f };
			size_t off = size_t(uvA->offset) + size_t(vi) * size_t(uvA->stride);
			auto uv = read_vec2_f32(*uvBytes, off);
			float u = uv.first;
			float v = 1.0f - uv.second; // you already flip V here, so tangent math matches what you render
			return { u, v };
			};

		// build tangent sums by walking triangles:
		if (can_build_tangents && mesh.indices.has_value()) {
			auto const& idx = mesh.indices.value();
			auto const& idxBytes = load_datafile(idx.src);

			// we assume triangle list for A2 (count is index count)
			uint32_t tri_count = mesh.count / 3;

			for (uint32_t t = 0; t < tri_count; ++t) {
				// read 3 indices:
				auto read_i = [&](uint32_t i) -> uint32_t {
					size_t idx_off = size_t(idx.offset);
					if (idx.format == VK_INDEX_TYPE_UINT16) idx_off += size_t(i) * 2;
					else idx_off += size_t(i) * 4;
					return read_index(idxBytes, idx_off, idx.format);
					};

				uint32_t i0 = read_i(t * 3 + 0);
				uint32_t i1 = read_i(t * 3 + 1);
				uint32_t i2 = read_i(t * 3 + 2);

				// safety (bad data shouldn't crash you)
				if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) continue;

				// positions:
				S72::vec3 p0 = read_pos(i0);
				S72::vec3 p1 = read_pos(i1);
				S72::vec3 p2 = read_pos(i2);

				// uvs:
				auto uv0 = read_uv(i0);
				auto uv1 = read_uv(i1);
				auto uv2 = read_uv(i2);

				float du1 = uv1.first - uv0.first;
				float dv1 = uv1.second - uv0.second;
				float du2 = uv2.first - uv0.first;
				float dv2 = uv2.second - uv0.second;

				// edge vectors in object space:
				S72::vec3 e1 = sub3(p1, p0);
				S72::vec3 e2 = sub3(p2, p0);

				// if UVs are degenerate, skip (prevents inf/nan tangents):
				float det = du1 * dv2 - dv1 * du2;
				if (std::fabs(det) < 1e-20f) continue;

				float r = 1.0f / det;

				// tangent points in the direction of +U on the surface
				S72::vec3 T = mul3s(sub3(mul3s(e1, dv2), mul3s(e2, dv1)), r);

				// bitangent points in the direction of +V on the surface
				S72::vec3 B = mul3s(sub3(mul3s(e2, du1), mul3s(e1, du2)), r);

				// add to each vertex in the triangle (later we normalize per vertex)
				tan_sum[i0] = add3(tan_sum[i0], T);
				tan_sum[i1] = add3(tan_sum[i1], T);
				tan_sum[i2] = add3(tan_sum[i2], T);

				bitan_sum[i0] = add3(bitan_sum[i0], B);
				bitan_sum[i1] = add3(bitan_sum[i1], B);
				bitan_sum[i2] = add3(bitan_sum[i2], B);
			}

			// finalize per-vertex tangents:
			for (uint32_t vi = 0; vi < vertex_count; ++vi) {
				S72::vec3 N = normalize3(read_nor(vi));
				S72::vec3 T = tan_sum[vi];

				// make T perpendicular to N (Gram-Schmidt)
				T = sub3(T, mul3s(N, dot3(N, T)));

				float tlen2 = dot3(T, T);
				if (tlen2 <= 0.0f) {
					// no good tangent -> keep default
					continue;
				}

				T = normalize3(T);

				// handedness sign:
				// if cross(N,T) points opposite the accumulated B, we flip sign
				S72::vec3 Bacc = bitan_sum[vi];
				float w = (dot3(cross3(N, T), Bacc) < 0.0f) ? -1.0f : +1.0f;

				tangents[vi] = Tan4{ T.x, T.y, T.z, w };
			}
		}

		auto emit_vertex = [&](uint32_t vtx_index) {
			size_t pos_off = size_t(posA->offset) + size_t(vtx_index) * size_t(posA->stride);
			S72::vec3 p = read_vec3_f32(posBytes, pos_off); //read positions

			//AABB accumulate (local space):update bounds
			range.local_min.x = std::min(range.local_min.x, p.x);
			range.local_min.y = std::min(range.local_min.y, p.y);
			range.local_min.z = std::min(range.local_min.z, p.z);

			range.local_max.x = std::max(range.local_max.x, p.x);
			range.local_max.y = std::max(range.local_max.y, p.y);
			range.local_max.z = std::max(range.local_max.z, p.z);



			 

			S72::vec3 n{ 0.0f, 0.0f, 1.0f };
			if (norA && norBytes) {
				size_t nor_off = size_t(norA->offset) + size_t(vtx_index) * size_t(norA->stride);
				n = read_vec3_f32(*norBytes, nor_off);
			}

			float s = 0.0f, t = 0.0f;
			if (uvA && uvBytes) {
				size_t uv_off = size_t(uvA->offset) + size_t(vtx_index) * size_t(uvA->stride);
				auto uv = read_vec2_f32(*uvBytes, uv_off);
				s = uv.first;
				t = 1.0f - uv.second; // flip V like you already do
			}

			Tan4 tan{ 1.0f, 0.0f, 0.0f, 1.0f };
			if (tanA && tanBytes) {
				size_t tan_off = size_t(tanA->offset) + size_t(vtx_index) * size_t(tanA->stride);
				auto loaded_tan = read_vec4_f32(*tanBytes, tan_off); // Now returns std::array
				tan = Tan4{ loaded_tan[0], loaded_tan[1], loaded_tan[2], loaded_tan[3] };
			}
			else if (vtx_index < tangents.size()) {
				tan = tangents[vtx_index]; // Fallback
			}

			// build one vertex and push it
			PosNorTexVertex v{};
			v.Position = { p.x, p.y, p.z };
			v.Normal = { n.x, n.y, n.z };
			v.TexCoord = { s, t };
			v.Tangent = { tan.x, tan.y, tan.z, tan.w };

			packed.emplace_back(v);
			};

		// Expand indices (or emit sequential vertices if non-indexed):
		if (mesh.indices.has_value()) {
			auto const& idx = mesh.indices.value();
			auto const& idxBytes = load_datafile(idx.src);

			// We assume triangles for now. If not triangles, we still just expand in given order.
			for (uint32_t i = 0; i < mesh.count; ++i) {
				size_t idx_off = size_t(idx.offset);
				if (idx.format == VK_INDEX_TYPE_UINT16) idx_off += size_t(i) * 2;
				else idx_off += size_t(i) * 4;

				uint32_t v = read_index(idxBytes, idx_off, idx.format);
				emit_vertex(v);
			}
		}
		else {
			for (uint32_t v = 0; v < mesh.count; ++v) emit_vertex(v);
		}

		range.count = uint32_t(packed.size()) - range.first;
		s72_mesh_to_range[mesh_ptr] = range;


		std::cout << "[A1-load] mesh '" << mesh.name << "' packed verts=" << range.count << "\n";
		std::cout << "[A1-load] mesh->range entries = " << s72_mesh_to_range.size() << "\n";

	}

	// Upload packed buffer:
	size_t bytes = packed.size() * sizeof(packed[0]);

	if (object_vertices.handle != VK_NULL_HANDLE) {
		rtg.helpers.destroy_buffer(std::move(object_vertices));
	}

	object_vertices = rtg.helpers.create_buffer(
		bytes,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	);

	if (!packed.empty()) {
		rtg.helpers.transfer_to_buffer(packed.data(), bytes, object_vertices);
	}

	std::cout << "[A1-load] S72 packed total verts=" << packed.size() << " bytes=" << bytes << "\n";
}

// --- frame time logging (headless benchmarking helper) ---



	if (scene_file.empty()) {
		{// create object vertices

			std::vector< PosNorTexVertex > vertices;

			auto append_obj = [&](const std::string& obj_path) -> ObjectVertices {
				ObjectVertices out;
				out.first = uint32_t(vertices.size());

				tinyobj::attrib_t attrib;
				std::vector<tinyobj::shape_t> shapes;
				std::string warn, err;

				bool ok = tinyobj::LoadObj(
					&attrib,
					&shapes,
					nullptr,          // no mtl
					&warn,
					&err,
					obj_path.c_str(),
					"data/",          // base dir
					true              // triangulate
				);

				if (!warn.empty()) std::cout << warn << std::endl;
				if (!err.empty())  std::cout << err << std::endl;
				if (!ok) {
					std::cout << "Failed to load " << obj_path << std::endl;
					out.count = 0;
					return out;
				}

				for (const auto& shape : shapes) {
					size_t index_offset = 0;
					for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
						int fv = shape.mesh.num_face_vertices[f];
						if (fv < 3) { index_offset += fv; continue; }

						for (int v = 0; v < 3; ++v) {
							tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

							float px = attrib.vertices[3 * idx.vertex_index + 0];
							float py = attrib.vertices[3 * idx.vertex_index + 1];
							float pz = attrib.vertices[3 * idx.vertex_index + 2];

							float nx = 0, ny = 0, nz = 1;
							if (idx.normal_index >= 0 && !attrib.normals.empty()) {
								nx = attrib.normals[3 * idx.normal_index + 0];
								ny = attrib.normals[3 * idx.normal_index + 1];
								nz = attrib.normals[3 * idx.normal_index + 2];
							}

							float ts = 0, tt = 0;
							if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
								ts = attrib.texcoords[2 * idx.texcoord_index + 0];
								tt = 1.0f - attrib.texcoords[2 * idx.texcoord_index + 1];

							}

							vertices.emplace_back(PosNorTexVertex{
								.Position{.x = px, .y = py, .z = pz},
								.Normal{.x = nx, .y = ny, .z = nz},
								.TexCoord{.s = ts, .t = tt},
								});
						}
						index_offset += fv;
					}
				}

				out.count = uint32_t(vertices.size()) - out.first;
				std::cout << "Loaded " << obj_path << " verts=" << out.count << "\n";
				return out;
				};

			chen_body_vertices = append_obj("data/chen_body.obj");
			chen_clothes_vertices = append_obj("data/chen_clothes.obj");
			chen_hairs_vertices = append_obj("data/chen_hairs.obj");
			chen_face_vertices = append_obj("data/chen_face.obj");
			chen_iris_vertices = append_obj("data/chen_iris.obj");
			chen_sword_vertices = append_obj("data/chen_sword.obj");





			{//A [-1,1]x[-1,1]x{0} quadrilateral:
				plane_vertices.first = uint32_t(vertices.size());

				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = -1.0f, .y = -1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 0.0f, .t = 0.0f },

					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = 1.0f, .y = -1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 1.0f, .t = 0.0f },

					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = -1.0f, .y = 1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 0.0f, .t = 1.0f },

					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = 1.0f, .y = 1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 1.0f, .t = 1.0f },
					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = -1.0f, .y = 1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f},
					.TexCoord{.s = 0.0f, .t = 1.0f },
					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = 1.0f, .y = -1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f},
					.TexCoord{.s = 1.0f, .t = 0.0f },
					});

				plane_vertices.count = uint32_t(vertices.size()) - plane_vertices.first;
			}

			{ // A torus:
				torus_vertices.first = uint32_t(vertices.size());

				//torus!
				//will parameterize with (u,v) where:
				// - u is angle around main axis (+z)
				// - v is angle around the tube

				constexpr float R1 = 0.75f; //main radius
				constexpr float R2 = 0.15F; //tube radius

				constexpr uint32_t U_STEPS = 20;
				constexpr uint32_t V_STEPS = 16;

				//texture repeats around the torus:
				constexpr float V_REPEATS = 2.0f;
				constexpr float U_REPEATS = int(V_REPEATS / R2 * R1 + 0.999f); //approximately square, 
				//rounded up

				auto emplace_vertex = [&](uint32_t ui, uint32_t vi) {
					//convert steps to angles:
					// (doing the mod since trig on 2 M_PI nay not exactly match 0)
					float ua = (ui % U_STEPS) / float(U_STEPS) * 2.0F * float(M_PI);
					float va = (vi % V_STEPS) / float(V_STEPS) * 2.0F * float(M_PI);

					vertices.emplace_back(PosNorTexVertex{
						.Position{
							.x = (R1 + R2 * std::cos(va)) * std::cos(ua),
							.y = (R1 + R2 * std::cos(va)) * std::sin(ua),
							.z = R2 * std::sin(va),
						},
						.Normal{
							.x = std::cos(va) * std::cos(ua),
							.y = std::cos(va) * std::sin(ua),
							.z = std::sin(ua),
						},
						.TexCoord{
							.s = ui / float(U_STEPS) * U_REPEATS,
							.t = vi / float(V_STEPS) * V_REPEATS,
						},
						});
					};

				for (uint32_t ui = 0; ui < U_STEPS; ++ui) {
					for (uint32_t vi = 0; vi < V_STEPS; ++vi) {
						emplace_vertex(ui, vi);
						emplace_vertex(ui + 1, vi);
						emplace_vertex(ui, vi + 1);

						emplace_vertex(ui, vi + 1);
						emplace_vertex(ui + 1, vi);
						emplace_vertex(ui + 1, vi + 1);


					}
				}

				torus_vertices.count = uint32_t(vertices.size()) - torus_vertices.first;
			}

			{ //A low-poly crystal (stylized gem) - simple + _ats nice:
				crystal_vertices.first = uint32_t(vertices.size());

				//local small vec type since PosNorTexVertex uses anonymous structs:
				struct P3 { float x, y, z; };

				constexpr uint32_t STEPS = 10; //8-12 looks good
				constexpr float R = 0.55f;   //ring radius
				constexpr float TOP = 1.0f;    //top point height
				constexpr float MID = 0.15f;   //ring height
				constexpr float BOT = -0.9f;   //bottom point height

				P3 top{ 0.0f, 0.0f, TOP };
				P3 bot{ 0.0f, 0.0f, BOT };

				auto add_tri = [&](P3 p0, P3 p1, P3 p2, P3 n,
					float s0, float t0,
					float s1, float t1,
					float s2, float t2) {
						vertices.emplace_back(PosNorTexVertex{
							.Position{.x = p0.x, .y = p0.y, .z = p0.z },
							.Normal{.x = n.x,  .y = n.y,  .z = n.z  },
							.TexCoord{.s = s0, .t = t0 },
							});
						vertices.emplace_back(PosNorTexVertex{
							.Position{.x = p1.x, .y = p1.y, .z = p1.z },
							.Normal{.x = n.x,  .y = n.y,  .z = n.z  },
							.TexCoord{.s = s1, .t = t1 },
							});
						vertices.emplace_back(PosNorTexVertex{
							.Position{.x = p2.x, .y = p2.y, .z = p2.z },
							.Normal{.x = n.x,  .y = n.y,  .z = n.z  },
							.TexCoord{.s = s2, .t = t2 },
							});
					};

				auto ring_point = [&](uint32_t i) -> P3 {
					float a = (i % STEPS) / float(STEPS) * 2.0f * float(M_PI);
					return P3{ R * std::cos(a), R * std::sin(a), MID };
					};

				for (uint32_t i = 0; i < STEPS; ++i) {
					P3 p0 = ring_point(i);
					P3 p1 = ring_point(i + 1);

					//outward-ish normals, good enough for now:
					float mx = (p0.x + p1.x) * 0.5f;
					float my = (p0.y + p1.y) * 0.5f;

					P3 nTop{ mx, my, 1.0f };
					P3 nBot{ mx, my, -1.0f };

					//top faces (CCW from outside)
					add_tri(p0, p1, top, nTop,
						i / float(STEPS), 0.0f,
						(i + 1) / float(STEPS), 0.0f,
						(i + 0.5f) / float(STEPS), 1.0f);

					//bottom faces (CCW from outside)
					add_tri(p1, p0, bot, nBot,
						(i + 1) / float(STEPS), 0.0f,
						i / float(STEPS), 0.0f,
						(i + 0.5f) / float(STEPS), 1.0f);
				}

				crystal_vertices.count = uint32_t(vertices.size()) - crystal_vertices.first;
			}







			size_t bytes = vertices.size() * sizeof(vertices[0]);

			object_vertices = rtg.helpers.create_buffer(
				bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			//copy data to buffer:
			rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
		}
	}

	{ //make some textures
		textures.reserve(256);

		// --- helpers needed by normal + PBR setup (MUST be before first use) ---

		auto load_png_normal_unorm = [&](std::string const& tex_path) -> uint32_t {
			auto it = normal_path_to_index.find(tex_path);
			if (it != normal_path_to_index.end()) return it->second;

			int w = 0, h = 0, n = 0;
			stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
			if (!pixels) {
				std::cout << "[A2-normal] failed to load normal map: " << tex_path << "\n";
				return 0; // fallback index
			}

			normal_maps.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
				VK_FORMAT_R8G8B8A8_UNORM, // linear, NOT SRGB
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, normal_maps.back());
			stbi_image_free(pixels);

			uint32_t idx = uint32_t(normal_maps.size() - 1);
			normal_path_to_index[tex_path] = idx;
			return idx;
			};

		auto make_solid_linear_unorm_texture = [&](float v01) -> uint32_t {
			v01 = std::max(0.0f, std::min(1.0f, v01));
			uint8_t u = uint8_t(std::round(v01 * 255.0f));

			uint32_t pixel = uint32_t(u) | (uint32_t(u) << 8) | (uint32_t(u) << 16) | (0xffu << 24);

			char key[64];
			std::snprintf(key, sizeof(key), "solid_lin_%u", unsigned(u));
			auto it = s72_texture_path_to_index.find(key);
			if (it != s72_texture_path_to_index.end()) return it->second;

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = 1, .height = 1 },
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(&pixel, sizeof(pixel), textures.back());

			uint32_t idx = uint32_t(textures.size() - 1);
			s72_texture_path_to_index[key] = idx;
			return idx;
			};

		auto load_png_texture_linear_unorm = [&](std::string const& tex_path) -> uint32_t {
			std::string key = "lin:" + tex_path;
			auto it = s72_texture_path_to_index.find(key);
			if (it != s72_texture_path_to_index.end()) return it->second;

			int w = 0, h = 0, n = 0;
			stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
			if (!pixels) {
				std::cout << "[A2-pbr] failed to load linear texture: " << tex_path << "\n";
				return 0;
			}

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
				VK_FORMAT_R8G8B8A8_UNORM, // linear (NOT SRGB)
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, textures.back());
			stbi_image_free(pixels);

			uint32_t idx = uint32_t(textures.size() - 1);
			s72_texture_path_to_index[key] = idx;
			return idx;
			};

		// set defaults ONCE (store into the outer variables)
		rough_default_idx = make_solid_linear_unorm_texture(1.0f); // white
		metal_default_idx = make_solid_linear_unorm_texture(0.0f); // black

		// --- end helpers ---

		// Make sure we always have a valid normal map at index 0 (flat normal).
		uint32_t flat_normal_idx = 0;
		if (normal_maps.empty()) {
			// RGBA = (128,128,255,255) => UNORM flat normal
			uint32_t pixel = 0xFFFF8080u;

			normal_maps.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = 1, .height = 1 },
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(&pixel, sizeof(pixel), normal_maps.back());
			flat_normal_idx = uint32_t(normal_maps.size() - 1);
		}
		else {
			// if you ever pre-fill normal_maps some other way, keep 0 as fallback
			flat_normal_idx = 0;
		}

		auto load_png_texture_srgb = [&](std::string const& tex_path) -> uint32_t {
			auto it = s72_texture_path_to_index.find(tex_path);
			if (it != s72_texture_path_to_index.end()) return it->second;

			int w = 0, h = 0, n = 0;
			stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
			if (!pixels) {
				std::cout << "[A1-show] failed to load texture: " << tex_path << "\n";
				return 0; // checker fallback
			}

			 

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, textures.back());
			stbi_image_free(pixels);

			uint32_t idx = uint32_t(textures.size() - 1);
			s72_texture_path_to_index[tex_path] = idx;
			return idx;
			};

		auto make_solid_srgb_texture = [&](S72::color const& c) -> uint32_t {
			// quantize to 8-bit and use as a cache key:
			auto to_u8 = [](float v) -> uint8_t {
				v = std::max(0.0f, std::min(1.0f, v));
				return uint8_t(std::round(v * 255.0f));
				};
			uint8_t r = to_u8(c.r), g = to_u8(c.g), b = to_u8(c.b);

			char key[64];
			std::snprintf(key, sizeof(key), "solid_%u_%u_%u", r, g, b);
			auto it = s72_texture_path_to_index.find(key);
			if (it != s72_texture_path_to_index.end()) return it->second;

			uint32_t pixel = uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (0xffu << 24);

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = 1, .height = 1 },
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(&pixel, sizeof(pixel), textures.back());

			uint32_t idx = uint32_t(textures.size() - 1);
			s72_texture_path_to_index[key] = idx;
			return idx;
			};

		 
		 

		// --- Build material->(albedo texture idx) mapping for S72 + prepare per-material set2 data:
		material_name_to_set2.clear();
		this->albedo_idx_per_set2.clear();
		this->normal_idx_per_set2.clear();

		if (use_s72_scene) {
			material_to_texture.clear();
			material_name_to_set2.clear();

			albedo_idx_per_set2.clear();
			normal_idx_per_set2.clear();
			roughness_idx_per_set2.clear();
			metalness_idx_per_set2.clear();

			for (auto const& mkv : scene.materials) {
				std::string const& mat_name = mkv.first;
				S72::Material const& mat = mkv.second;

				// ---- albedo index ----
				uint32_t albedo_idx = 0;

				if (auto const* lam = std::get_if<S72::Material::Lambertian>(&mat.brdf)) {
					if (std::holds_alternative<S72::color>(lam->albedo)) {
						albedo_idx = make_solid_srgb_texture(std::get<S72::color>(lam->albedo));
					}
					else {
						S72::Texture* t = std::get<S72::Texture*>(lam->albedo);
						if (t) { // REMOVED type == flat check
							albedo_idx = load_png_texture_srgb(t->path);
						}
					}
				}
				else if (auto const* pbr = std::get_if<S72::Material::PBR>(&mat.brdf)) {
					if (std::holds_alternative<S72::color>(pbr->albedo)) {
						albedo_idx = make_solid_srgb_texture(std::get<S72::color>(pbr->albedo));
					}
					else {
						S72::Texture* t = std::get<S72::Texture*>(pbr->albedo);
						if (t) { // REMOVED type == flat check
							albedo_idx = load_png_texture_srgb(t->path);
						}
					}
				}

				// ---- normal index ----
				uint32_t normal_idx = flat_normal_idx;
				if (mat.normal_map) {
					uint32_t loaded = load_png_normal_unorm(mat.normal_map->path);
					if (loaded != 0 || flat_normal_idx == 0) normal_idx = loaded;
				}

				// ---- roughness + metalness index (PBR only, else defaults) ----
				uint32_t rough_idx = rough_default_idx;
				uint32_t metal_idx = metal_default_idx;

				if (auto const* pbr = std::get_if<S72::Material::PBR>(&mat.brdf)) {

					// roughness:
					if (std::holds_alternative<float>(pbr->roughness)) {
						rough_idx = make_solid_linear_unorm_texture(std::get<float>(pbr->roughness));
					}
					else {
						S72::Texture* t = std::get<S72::Texture*>(pbr->roughness);
						if (t) { // REMOVED type == flat check
							rough_idx = load_png_texture_linear_unorm(t->path);
						}
					}

					// metalness:
					if (std::holds_alternative<float>(pbr->metalness)) {
						metal_idx = make_solid_linear_unorm_texture(std::get<float>(pbr->metalness));
					}
					else {
						S72::Texture* t = std::get<S72::Texture*>(pbr->metalness);
						if (t) { // REMOVED type == flat check
							metal_idx = load_png_texture_linear_unorm(t->path);
						}
					}
				}

				// assign set2 index for this material:
				uint32_t set2_index = uint32_t(albedo_idx_per_set2.size());
				material_name_to_set2[mat_name] = set2_index;

				albedo_idx_per_set2.push_back(albedo_idx);
				normal_idx_per_set2.push_back(normal_idx);
				roughness_idx_per_set2.push_back(rough_idx);
				metalness_idx_per_set2.push_back(metal_idx);

				// keep old mapping if other code still uses it:
				material_to_texture[&mat] = albedo_idx;
			}

			std::cout << "[A2-pbr] materials prepared for set2: " << albedo_idx_per_set2.size() << "\n";
		}


		{ //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
			//actually make the texture:
			uint32_t size = 128;
			std::vector< uint32_t > data;
			data.reserve(size * size);
			for (uint32_t y = 0; y < size; ++y) {
				float fy = (y + 0.5f) / float(size);
				for (uint32_t x = 0; x < size; ++x) {
					float fx = (x + 0.5f) / float(size);
					//highlight the origin:
					if (fx < 0.05f && fy < 0.05f) data.emplace_back(0xff0000ff); //red
					else if ((fx < 0.5f) == (fy < 0.5f)) data.emplace_back(0xff444444); //dark grey
					else data.emplace_back(0xffbbbbbb); //light grey
				}
			}
			assert(data.size() == size * size);

			//make a place for the texture to live on the GPU
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = size , .height = size }, //size of image
				VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data (in this case, linearly-encoded 8-bit RGBA)
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
				Helpers::Unmapped
			));

			//transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		}

			{ //texture 1 will be a classic 'xor' texture
				//actually make the texture:
				uint32_t size = 256;
				std::vector< uint32_t > data;
				data.reserve(size* size);
				for (uint32_t y = 0; y < size; ++y) {
					for (uint32_t x = 0; x < size; ++x) {
						uint8_t r = uint8_t(x) ^ uint8_t(y);
						uint8_t g = uint8_t(x + 128) ^ uint8_t(y);
						uint8_t b = uint8_t(x) ^ uint8_t(y + 27);
						uint8_t a = 0xff;
						data.emplace_back(uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24));
					}
				}
				assert(data.size() == size * size);

				//make a place for the texture to live on the GPU:
				textures.emplace_back(rtg.helpers.create_image(
					VkExtent2D{ .width = size , .height = size }, //size of image
					VK_FORMAT_R8G8B8A8_SRGB, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
					VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
					Helpers::Unmapped
				));

			//transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
			}


			if (scene_file.empty()) {
				auto load_png_texture = [&](std::string const& tex_path) -> uint32_t {
					int w = 0, h = 0, n = 0;
					stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
					if (!pixels) {
						std::cout << "stb_image failed to load: " << tex_path << std::endl;
						return 0; //fallback to texture 0
					}

					textures.emplace_back(rtg.helpers.create_image(
						VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
						VK_FORMAT_R8G8B8A8_SRGB,
						VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						Helpers::Unmapped
					));

					rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, textures.back());
					stbi_image_free(pixels);

					uint32_t idx = uint32_t(textures.size() - 1);
					std::cout << "Loaded texture[" << idx << "]: " << tex_path << " (" << w << "x" << h << ")\n";
					return idx;
					};




				// load textures and store the actual indices:
				tex_body = load_png_texture("data/chen_body.png");
				tex_clothes = load_png_texture("data/chen_clothes.png");
				tex_hair = load_png_texture("data/chen_hair.png");
				tex_sword = load_png_texture("data/chen_sword.png");
				tex_iris = load_png_texture("data/chen_iris.png");
				tex_face = load_png_texture("data/chen_face.png");
			}

		
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const& image : textures) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.flags = 0,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = image.format,
				// .components sets swizzling and is fine when zero-initialized
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkImageView image_view = VK_NULL_HANDLE;
			VK(vkCreateImageView(rtg.device, &create_info, nullptr, &image_view));

			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{ // make image views for normal maps
		normal_map_views.reserve(normal_maps.size());
		for (Helpers::AllocatedImage const& image : normal_maps) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = image.format,
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkImageView view = VK_NULL_HANDLE;
			VK(vkCreateImageView(rtg.device, &create_info, nullptr, &view));
			normal_map_views.emplace_back(view);
		}
	}

	 

	 

	 

	{ //make a sampler for the textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f, //doesn't matter if anisotropy isn't enabled
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS, //doesn't matter if compare isn't enabled
			.minLod = 0.0f,
			.maxLod = 9.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK(vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler));

		
		  
	}

	// --- A2-pbr: load BRDF LUT png (2D) ---
	{
		std::string lut_path = "data/brdf_lut.png";  

		int w = 0, h = 0, n = 0;
		stbi_uc* pixels = stbi_load(lut_path.c_str(), &w, &h, &n, 4);
		if (!pixels) {
			std::cout << "[A2-pbr] failed to load BRDF LUT: " << lut_path << "\n";
			has_brdf_lut = false;
		}
		else {
			brdf_lut = rtg.helpers.create_image(
				VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
				VK_FORMAT_R8G8B8A8_UNORM, // because you wrote RG into RGBA8
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, brdf_lut);
			stbi_image_free(pixels);

			VkImageViewCreateInfo view_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = brdf_lut.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = brdf_lut.format,
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VK(vkCreateImageView(rtg.device, &view_info, nullptr, &brdf_lut_view));

			has_brdf_lut = true;
			std::cout << "[A2-pbr] loaded BRDF LUT: " << lut_path << " (" << w << "x" << h << ")\n";
			
		}
	}
	

	{ //create the texture descriptor pool (set2 per material for S72, else per texture)
		uint32_t set2_count = 0;
		if (use_s72_scene) {
			set2_count = uint32_t(scene.materials.size());
		}
		else {
			set2_count = uint32_t(textures.size());
		}

		const uint32_t extra_sets = 5;
		const uint32_t extra_samplers = 10;

		std::array<VkDescriptorPoolSize, 1> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				// 2 bindings for Lambertian + 4 bindings for PBR = 6 per material
				.descriptorCount = 6 * set2_count + extra_samplers,
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			// We are allocating TWO sets per material now, so multiply by 2
			.maxSets = 2 * set2_count + extra_sets,
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool));
	}

	{ //allocate and write set2_TEXTURE descriptor sets (4 bindings)
		VkDescriptorSetLayout set2_layout = objects_pipeline.set2_TEXTURE; // unified layout

		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = texture_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &set2_layout,
		};

	 

		if (use_s72_scene) {
			// 1. Allocate Lambertian Sets (2 bindings)
			material_descriptors_lam.assign(scene.materials.size(), VK_NULL_HANDLE);
			VkDescriptorSetLayout lam_layout = objects_pipeline.set2_TEXTURE;
			VkDescriptorSetAllocateInfo alloc_lam{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = texture_descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lam_layout,
			};

			// 2. Allocate PBR Sets (4 bindings)
			material_descriptors_pbr.assign(scene.materials.size(), VK_NULL_HANDLE);
			VkDescriptorSetLayout pbr_layout = pbr_pipeline.set2_TEXTURE;
			VkDescriptorSetAllocateInfo alloc_pbr{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = texture_descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &pbr_layout,
			};

			for (size_t i = 0; i < scene.materials.size(); ++i) {
				VK(vkAllocateDescriptorSets(rtg.device, &alloc_lam, &material_descriptors_lam[i]));
				VK(vkAllocateDescriptorSets(rtg.device, &alloc_pbr, &material_descriptors_pbr[i]));
			}

			std::vector<VkWriteDescriptorSet> writes;
			std::vector<VkDescriptorImageInfo> infos(scene.materials.size() * 6); // 2 Lam + 4 PBR

			for (size_t i = 0; i < scene.materials.size(); ++i) {
				uint32_t albedo_idx = albedo_idx_per_set2[i];
				uint32_t normal_idx = normal_idx_per_set2[i];
				uint32_t rough_idx = (i < roughness_idx_per_set2.size()) ? roughness_idx_per_set2[i] : rough_default_idx;
				uint32_t metal_idx = (i < metalness_idx_per_set2.size()) ? metalness_idx_per_set2[i] : metal_default_idx;

				// Fill Image Infos
				infos[6 * i + 0] = VkDescriptorImageInfo{ texture_sampler, texture_views[albedo_idx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[6 * i + 1] = VkDescriptorImageInfo{ texture_sampler, normal_map_views[normal_idx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

				infos[6 * i + 2] = VkDescriptorImageInfo{ texture_sampler, texture_views[albedo_idx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[6 * i + 3] = VkDescriptorImageInfo{ texture_sampler, normal_map_views[normal_idx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[6 * i + 4] = VkDescriptorImageInfo{ texture_sampler, texture_views[rough_idx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[6 * i + 5] = VkDescriptorImageInfo{ texture_sampler, texture_views[metal_idx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

				// Lambertian Writes
				writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, material_descriptors_lam[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[6 * i + 0], nullptr, nullptr });
				writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, material_descriptors_lam[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[6 * i + 1], nullptr, nullptr });

				// PBR Writes
				writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, material_descriptors_pbr[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[6 * i + 2], nullptr, nullptr });
				writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, material_descriptors_pbr[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[6 * i + 3], nullptr, nullptr });
				writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, material_descriptors_pbr[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[6 * i + 4], nullptr, nullptr });
				writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, material_descriptors_pbr[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[6 * i + 5], nullptr, nullptr });
			}

			vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
		}
		else {
			// one set2 per texture (non-s72 mode):
			texture_descriptors.assign(textures.size(), VK_NULL_HANDLE);
			for (VkDescriptorSet& set : texture_descriptors) {
				VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &set));
			}

			std::vector<VkDescriptorImageInfo> infos(textures.size() * 4);
			std::vector<VkWriteDescriptorSet> writes(textures.size() * 4);

			for (size_t i = 0; i < textures.size(); ++i) {
				VkImageView albedo_view = texture_views[i];
				VkImageView normal_view = normal_map_views.empty() ? texture_views[i] : normal_map_views[0]; // flat normal if available

				VkImageView rough_view = texture_views[rough_default_idx];
				VkImageView metal_view = texture_views[metal_default_idx];

				infos[4 * i + 0] = VkDescriptorImageInfo{ texture_sampler, albedo_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[4 * i + 1] = VkDescriptorImageInfo{ texture_sampler, normal_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[4 * i + 2] = VkDescriptorImageInfo{ texture_sampler, rough_view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
				infos[4 * i + 3] = VkDescriptorImageInfo{ texture_sampler, metal_view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

				writes[4 * i + 0] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, texture_descriptors[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[4 * i + 0], nullptr, nullptr };
				writes[4 * i + 1] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, texture_descriptors[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[4 * i + 1], nullptr, nullptr };
				writes[4 * i + 2] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, texture_descriptors[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[4 * i + 2], nullptr, nullptr };
				writes[4 * i + 3] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, texture_descriptors[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos[4 * i + 3], nullptr, nullptr };
			}

			vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
		}
	}

	{
		{ // A2 background descriptor set (ALWAYS allocate; use env if available else texture 0)
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = texture_descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &background_pipeline.set_layout,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &background_pipeline.set));

			VkDescriptorImageInfo bg_info{
				.sampler = env_sampler,
				.imageView = env_cubemap_view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkWriteDescriptorSet write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = background_pipeline.set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &bg_info,
			};

			vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);

			// --- A2-diffuse: allocate/write set=3 lambertian env descriptor ---
			{
				VkDescriptorSetAllocateInfo lam_alloc_info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = texture_descriptor_pool,
					.descriptorSetCount = 1,
					.pSetLayouts = &objects_pipeline.set3_EnvLambertian,
				};
				VK(vkAllocateDescriptorSets(rtg.device, &lam_alloc_info, &env_lambertian_descriptors));

				// If we failed to load lambertian, fall back to env_cubemap_view (or dummy env)
				VkImageView lam_view =
					(has_env_lambertian && env_lambertian_cubemap_view != VK_NULL_HANDLE)
					? env_lambertian_cubemap_view
					: env_cubemap_view;

				VkDescriptorImageInfo lam_info{
					.sampler = env_sampler,
					.imageView = lam_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				VkWriteDescriptorSet lam_write{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = env_lambertian_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &lam_info,
				};

				vkUpdateDescriptorSets(rtg.device, 1, &lam_write, 0, nullptr);
			}

			// --- A2-pbr: allocate/write set=3 PBR env descriptor (binding0 lambertian, binding1 ggx, binding2 brdf lut) ---
			{
				VkDescriptorSetAllocateInfo pbr_alloc_info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = texture_descriptor_pool,
					.descriptorSetCount = 1,
					.pSetLayouts = &pbr_pipeline.set3_EnvPBR, // PBR pipeline's set3 layout (3 bindings)
				};
				VK(vkAllocateDescriptorSets(rtg.device, &pbr_alloc_info, &env_pbr_descriptors));

				// binding 0: lambertian (irradiance)
				VkImageView lam_view =
					(has_env_lambertian && env_lambertian_cubemap_view != VK_NULL_HANDLE)
					? env_lambertian_cubemap_view
					: env_cubemap_view;

				// binding 1: ggx (prefiltered specular)
				VkImageView ggx_view =
					(has_env_ggx && env_ggx_cubemap_view != VK_NULL_HANDLE)
					? env_ggx_cubemap_view
					: env_cubemap_view; // safe fallback

				// binding 2: brdf lut (2D)
				VkImageView lut_view =
					(has_brdf_lut && brdf_lut_view != VK_NULL_HANDLE)
					? brdf_lut_view
					: VK_NULL_HANDLE;

				VkDescriptorImageInfo lam_info{
					.sampler = env_sampler,
					.imageView = lam_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				VkDescriptorImageInfo ggx_info{
					.sampler = env_sampler,
					.imageView = ggx_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

			 
				assert(lut_view != VK_NULL_HANDLE);

				VkDescriptorImageInfo lut_info{
					.sampler = env_sampler,
					.imageView = lut_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				std::array<VkWriteDescriptorSet, 3> writes{
					VkWriteDescriptorSet{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = env_pbr_descriptors,
						.dstBinding = 0,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &lam_info,
					},
					VkWriteDescriptorSet{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = env_pbr_descriptors,
						.dstBinding = 1,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &ggx_info,
					},
					VkWriteDescriptorSet{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = env_pbr_descriptors,
						.dstBinding = 2,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &lut_info,
					},
				};

				vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
			}
		}



		 
	}

	

	// --- frame time logging 
	//ft_log_enabled = rtg.configuration.headless;
	//if (ft_log_enabled) {
		//ft_logger.start("frametimes.csv");
	//}
}


Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)

	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

	// Samplers
	if (env_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(rtg.device, env_sampler, nullptr);
		env_sampler = VK_NULL_HANDLE;
	}

	if (shadow_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(rtg.device, shadow_sampler, nullptr);
		shadow_sampler = VK_NULL_HANDLE;
	}
	// GGX Specular Cubemap
	if (env_ggx_cubemap_view != VK_NULL_HANDLE) {
		vkDestroyImageView(rtg.device, env_ggx_cubemap_view, nullptr);
		env_ggx_cubemap_view = VK_NULL_HANDLE;
	}
	rtg.helpers.destroy_image(std::move(env_ggx_cubemap));

	// Lambertian Irradiance Cubemap
	if (env_lambertian_cubemap_view != VK_NULL_HANDLE) {
		vkDestroyImageView(rtg.device, env_lambertian_cubemap_view, nullptr);
		env_lambertian_cubemap_view = VK_NULL_HANDLE;
	}
	rtg.helpers.destroy_image(std::move(env_lambertian_cubemap));

	// BRDF LUT
	if (brdf_lut_view != VK_NULL_HANDLE) {
		vkDestroyImageView(rtg.device, brdf_lut_view, nullptr);
		brdf_lut_view = VK_NULL_HANDLE;
	}
	rtg.helpers.destroy_image(std::move(brdf_lut));

	// Fallback/Dummy Cubemap
	if (env_cubemap_view != VK_NULL_HANDLE) {
		vkDestroyImageView(rtg.device, env_cubemap_view, nullptr);
		env_cubemap_view = VK_NULL_HANDLE;
	}
	rtg.helpers.destroy_image(std::move(env_cubemap));

	// Normal Maps
	for (VkImageView& view : normal_map_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	normal_map_views.clear();

	for (auto& nm : normal_maps) {
		rtg.helpers.destroy_image(std::move(nm));
	}
	normal_maps.clear();

	// --- Standard Texture Cleanup ---

	if (texture_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

	for (VkImageView& view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (auto& texture : textures) {
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

	// --- Buffer & Pool Cleanup ---

	rtg.helpers.destroy_buffer(std::move(object_vertices));

	if (texture_descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = VK_NULL_HANDLE;
	}

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
	}

	for (Workspace& workspace : workspaces) {
		if (workspace.command_buffer != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(rtg.device, command_pool, 1, &workspace.command_buffer);
		}
		rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		rtg.helpers.destroy_buffer(std::move(workspace.World));
		rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
	}
	workspaces.clear();

	// --- Pipeline & Pass Cleanup ---

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);
	pbr_pipeline.destroy(rtg);
	mirror_pipeline.destroy(rtg);

	if (command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}

	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces) {


		 
		if (workspace.command_buffer != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(rtg.device,  command_pool, 1, &workspace.command_buffer);
			workspace.command_buffer = VK_NULL_HANDLE;
		}

		//cleaning up per-workspace lines buffers in Tutorial::~Tutorial
		if (workspace.lines_vertices_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if (workspace.lines_vertices.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}

		if (workspace.Camera_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}

		//Camera_descriptors freed when pool is destroyed.

		if (workspace.World_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		}
		if (workspace.World.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World));
		}
		//World_descriptors freed when pool is destroyed.

		

		if (workspace.Transforms_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
		//Transforms_descriptor freed when pool is destroyed

	}

	
	workspaces.clear();

	if (descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
		//(this also frees the descriptor sets allocated from the pool)
	}

	//destroy pipeline (sequenced in opposite order of construction)

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);
	pbr_pipeline.destroy(rtg);
	mirror_pipeline.destroy(rtg);


	//destroy command pool
	if (command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE) { //cleanup
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}
}

void Tutorial::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain) {
	 //TODO: clean up existing framebuffers
	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}
	//Allocate depth image for framebuffers to share:
	swapchain_depth_image = rtg.helpers.create_image(
		swapchain.extent,
		depth_format,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	);

	{//create an image view of the depth image:
		VkImageViewCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchain_depth_image.handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
		};

		VK(vkCreateImageView(rtg.device, &create_info, nullptr, &swapchain_depth_image_view));
	}

	//create framebuffers pointing to each swapchain image view and the shared depth image view
	//framebuffers for each swapchain image:
	swapchain_framebuffers.assign(swapchain.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain.image_views.size(); ++i) {
		std::array< VkImageView, 2 > attachments{
			swapchain.image_views[i],
			swapchain_depth_image_view,
		};
		VkFramebufferCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = render_pass,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.width = swapchain.extent.width,
			.height = swapchain.extent.height,
			.layers = 1,
		};

		VK(vkCreateFramebuffer(rtg.device, &create_info, nullptr, &swapchain_framebuffers[i]));
	}
}

void Tutorial::destroy_framebuffers() {
	 
	for (VkFramebuffer& framebuffer : swapchain_framebuffers) {
		assert(framebuffer != VK_NULL_HANDLE);
		vkDestroyFramebuffer(rtg.device, framebuffer, nullptr);
		framebuffer = VK_NULL_HANDLE;
	}
	swapchain_framebuffers.clear();

	assert(swapchain_depth_image_view != VK_NULL_HANDLE);
	vkDestroyImageView(rtg.device, swapchain_depth_image_view, nullptr);
	swapchain_depth_image_view = VK_NULL_HANDLE;

	rtg.helpers.destroy_image(std::move(swapchain_depth_image));
}


void Tutorial::render(RTG& rtg_, RTG::RenderParams const& render_params) {
	//assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	//get more convenient names for the current workspace and target framebuffer:
	Workspace& workspace = workspaces[render_params.workspace_index];
	VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	//reset the command buffer (clear old commands):
	VK(vkResetCommandBuffer(workspace.command_buffer, 0));
	{//begin recording
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, //set to the proper for this structure
			//.pNext set to null by zero-initialization!
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, //will record again every submit
		};
		VK(vkBeginCommandBuffer(workspace.command_buffer, &begin_info));
	}

	if (!lines_vertices.empty()) {//upload lines vertices:
		//[re-]allocate lines buffers if needed:
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if (workspace.lines_vertices_src.handle == VK_NULL_HANDLE ||
			workspace.lines_vertices_src.size < needed_bytes) {
			//round to next multiple of 4k to avoid re-allocating continuously if vertex count grows slowly:
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			if (workspace.lines_vertices_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if (workspace.lines_vertices.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
			}
			//actual memory allocation
			workspace.lines_vertices_src = rtg.helpers.create_buffer( //use staging buffer
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible memory,
				//coherent (no special sync needed)
				Helpers::Mapped //get a pointer to the memory
			);
			workspace.lines_vertices = rtg.helpers.create_buffer(//
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as vertex buffer
				//also going to have GPU into this memory 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
				Helpers::Unmapped //don't get a pointer to the memory
			);

			std::cout << "Re-allocated lines buffers to " << new_bytes << "bytes." << std::endl;

		}

		assert(workspace.lines_vertices_src.size == workspace.lines_vertices.size);
		assert(workspace.lines_vertices_src.size >= needed_bytes);

		//host-side copy into lines_vertices_src;
		assert(workspace.lines_vertices_src.allocation.mapped);
		//helper allocatin data member function is to account for any offset in the allocation when getting a
		//pointer to start the allocation's mapped memory
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);

		//device-side copy from lines_vertices_src -> lines_vertices;
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.
			handle, 1, &copy_region);
	}

	{//upload camera info:
		LinesPipeline::Camera camera{
			.CLIP_FROM_WORLD = CLIP_FROM_WORLD
		};
		assert(workspace.Camera_src.size == sizeof(camera));

		//host-side copy into Camera_src:
		memcpy(workspace.Camera_src.allocation.data(), &camera, sizeof(camera));

		//add device-side copy from Camera_src -> Camera:
		assert(workspace.Camera_src.size == workspace.Camera.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Camera_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Camera_src.handle,
			workspace.Camera.handle, 1, &copy_region);
	}

	{ //upload world info:
		assert(workspace.World_src.size == sizeof(world));

		//host-side copy into World_src:
		memcpy(workspace.World_src.allocation.data(), &world, sizeof(world));

		//add device-side copy from World_src -> World:
		assert(workspace.World_src.size == workspace.World.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.World_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.World_src.handle, workspace.World.handle, 1, &copy_region);
	}

	/* {//upload light info
		//upload light data to gpu
		std::vector<GPULight> gpu_lights;

		for (auto const& l : loaded_lights) {
			GPULight g{};

			g.position = { l.world_position.x, l.world_position.y, l.world_position.z, float(l.type) };

			g.direction = { l.world_direction.x, l.world_direction.y, l.world_direction.z, 0.0f };

			g.tint = { l.tint.x, l.tint.y, l.tint.z, l.shadow };

			g.params = { l.radius, l.power, l.limit, l.fov };

			gpu_lights.push_back(g);
		}

		std::cout << "uploading lights count: " << loaded_lights.size() << std::endl;
		std::cout << "sizeof(GPULight): " << sizeof(GPULight) << std::endl;
		memcpy(
			(void*)workspace.Lights.allocation.data(),
			(const void*)gpu_lights.data(),
			gpu_lights.size() * sizeof(GPULight)
		);
	
	}*/

	if (workspace.Lights.handle != VK_NULL_HANDLE) {
		//upload light info
		std::vector<GPULight> gpu_lights;

		for (auto const& l : loaded_lights) {
			GPULight g{};

			//g.position = { l.world_position.x, l.world_position.y, l.world_position.z, float(l.type) };
			float type_value = 0.0f;
			if (l.type == LoadedLight::Type::Sun) type_value = 0.0f;
			else if (l.type == LoadedLight::Type::Sphere) type_value = 1.0f;
			else if (l.type == LoadedLight::Type::Spot) type_value = 2.0f;

			g.position = { l.world_position.x, l.world_position.y, l.world_position.z, type_value };
			g.direction = { l.world_direction.x, l.world_direction.y, l.world_direction.z, l.blend };
			g.tint = { l.tint.x, l.tint.y, l.tint.z, l.shadow };
			if (l.type == LoadedLight::Type::Sun) {
				g.params = { l.angle, l.strength, 0.0f, 0.0f };
			}
			else {
				g.params = { l.radius, l.power, l.limit, l.fov };
			}

			gpu_lights.push_back(g);
		}

		//std::cout << "uploading lights count: " << loaded_lights.size() << std::endl;
		//std::cout << "sizeof(GPULight): " << sizeof(GPULight) << std::endl;

		memcpy(
			(void*)workspace.Lights.allocation.data(),
			(const void*)gpu_lights.data(),
			gpu_lights.size() * sizeof(GPULight)
		);
	}


	if (!object_instances.empty()) { //upload object transforms:
		size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		if (workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size <
			needed_bytes) {
			//round to next multiple of 4k to avoid re-allocating continuously 
			// if vertex count grows slowly:
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			if (workspace.Transforms_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if (workspace.Transforms.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}
			//actual memory allocation
			workspace.Transforms_src = rtg.helpers.create_buffer( //use staging buffer
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible memory,
				//coherent (no special sync needed)
				Helpers::Mapped //get a pointer to the memory
			);
			workspace.Transforms = rtg.helpers.create_buffer(//
				new_bytes,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as vertex buffer
				//also going to have GPU into this memory 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
				Helpers::Unmapped //don't get a pointer to the memory
			);

			//update the descriptor set:
			VkDescriptorBufferInfo Transforms_info{
				.buffer = workspace.Transforms.handle,
				.offset = 0,
				.range = workspace.Transforms.size,
			};

			std::array<VkWriteDescriptorSet, 1> writes{
	VkWriteDescriptorSet{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = workspace.Transforms_descriptors,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &Transforms_info,
	},
			};

			vkUpdateDescriptorSets(
				rtg.device,
				uint32_t(writes.size()),
				writes.data(),
				0,
				nullptr
			);

			std::cout << "Re-allocated object transforms buffers to " << new_bytes << "bytes."
				<< std::endl;
		}

		assert(workspace.Transforms_src.size == workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);

		{ //copy transforms into Transforms_src:
			assert(workspace.Transforms_src.allocation.mapped);
			ObjectsPipeline::Transform* out = reinterpret_cast<ObjectsPipeline::Transform*>
				(workspace.Transforms_src.allocation.data()); // Strict aliasing violation, but it doesn't matter
			for (ObjectInstance const& inst : object_instances) {
				*out = inst.transform;
				++out;
			}
		}

		//device-side copy from Transform_src -> Transform;
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Transforms_src.handle,
			workspace.Transforms.handle, 1, &copy_region);

	}


	{//Memory barrier to make sure copies complete before rendering happens;
		VkMemoryBarrier memory_barrier{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
			| VK_ACCESS_UNIFORM_READ_BIT
			| VK_ACCESS_SHADER_READ_BIT,
		};

		vkCmdPipelineBarrier(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, //srcStageMask
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, //dependencyFlags
			1, &memory_barrier, //memoryBarriers (count, data)
			0, nullptr, //bufferMemoryBarriers (count, data)
			0, nullptr //imageMemoryBarriers (count, data)
		);
	}

	 

	//GPU commands here:

 
	for (size_t li = 0; li < shadow_spot_lights.size(); ++li) {

		auto m = make_spot_light_matrix(*shadow_spot_lights[li]);

		/*std::cout << "LIGHT MATRIX:\n";
		for (int k = 0; k < 16; k++) {
			std::cout << m[k] << " ";
			if ((k % 4) == 3) std::cout << "\n";
		}*/

		VkClearValue clear{};
		clear.depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = shadow_render_pass,
			.framebuffer = shadow_framebuffers[li],
			.renderArea{
				.offset = {0, 0},
				.extent = shadow_maps[li].extent,
			},
			.clearValueCount = 1,
			.pClearValues = &clear,
		};

		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(
			workspace.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			shadow_pipeline.handle
		);

		vkCmdSetDepthBias(workspace.command_buffer, 0.0f, 0.0f, 0.0f);
		 

		VkViewport vp{
	0.0f,
	0.0f,
	float(shadow_maps[li].extent.width),
	float(shadow_maps[li].extent.height),
	0.0f,
	1.0f
		};

		VkRect2D sc{
			{0, 0},
			shadow_maps[li].extent
		};

		vkCmdSetViewport(workspace.command_buffer, 0, 1, &vp);
		vkCmdSetScissor(workspace.command_buffer, 0, 1, &sc);

		VkBuffer vb = object_vertices.handle;
		VkDeviceSize off = 0;
		vkCmdBindVertexBuffers(workspace.command_buffer, 0, 1, &vb, &off);

		vkCmdBindDescriptorSets(
			workspace.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			shadow_pipeline.layout,
			0,
			1,
			&workspace.Transforms_descriptors,
			0,
			nullptr
		);

		//std::cout << "shadow draw count: " << object_instances.size() << "\n";

		/*if (!object_instances.empty()) {
			std::cout << "first shadow verts: "
				<< object_instances[0].vertices.first << " "
				<< object_instances[0].vertices.count << "\n";
		}*/

		for (uint32_t i = 0; i < object_instances.size(); ++i) {
			auto const& inst = object_instances[i];
			Tutorial::ShadowPush push{};
			push.LIGHT_CLIP_FROM_WORLD = m;
			push.OBJECT_INDEX = int32_t(i);

			vkCmdPushConstants(
				workspace.command_buffer,
				shadow_pipeline.layout,
				VK_SHADER_STAGE_VERTEX_BIT,
				0,
				sizeof(Tutorial::ShadowPush),
				&push
			);
			vkCmdDraw(
				workspace.command_buffer,
				inst.vertices.count,
				1,
				inst.vertices.first,
				i
			);
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}
	
	{//render pass
		std::array< VkClearValue, 2 > clear_values{
			VkClearValue{.color{.float32{1.0f, 0.85f, 0.90f, 1.0f}
}},
			VkClearValue{.depthStencil{.depth = 1.0f, .stencil = 0}},
		};

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			 },

			.clearValueCount = uint32_t(clear_values.size()),
.pClearValues = clear_values.data(),
		};

		
		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);
		vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);

		//  run pipelines here:
 
		{
			// ---    background pass ---
			if (background_pipeline.set != VK_NULL_HANDLE) {

				vkCmdBindPipeline(
					workspace.command_buffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					background_pipeline.handle
				);

				// bind background descriptor set (set 0)
				vkCmdBindDescriptorSets(
					workspace.command_buffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					background_pipeline.layout,   //  
					0,                            // firstSet
					1,
					&background_pipeline.set,
					0,
					nullptr
				);

				// push constants
				Tutorial::BackgroundPipeline::Push push{};
				push.time = time;

				// Use the Tutorial members (do NOT redeclare locals named the same)
				mat4 inv_proj = mat4_inverse(this->CLIP_FROM_VIEW);

				// VIEW_FROM_WORLD is world->view. Inverse gives view->world.
				mat4 world_from_view = mat4_inverse(this->VIEW_FROM_WORLD);

				int tone_op = 0; // 0 = linear
				if (rtg.configuration.tone_map == "reinhard") tone_op = 1;



				//   mat4 is a flat 16-float array.
				// Translation lives at indices 12,13,14  ( mul() and mat4_translate()).
				// Zero translation so we keep rotation-only for the sky/background.
				world_from_view[12] = 0.0f;
				world_from_view[13] = 0.0f;
				world_from_view[14] = 0.0f;

				push.inv_proj = inv_proj;
				push.inv_view_rot = world_from_view;
				push.exposure = rtg.configuration.exposure; //   exposure in stops
				push.tone_op = tone_op;                 // tone map operator id

				vkCmdPushConstants(
					workspace.command_buffer,
					background_pipeline.layout,
					VK_SHADER_STAGE_FRAGMENT_BIT,
					0,
					sizeof(Tutorial::BackgroundPipeline::Push),
					&push
				);

				//  
				vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);
				vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);

				vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);

			}

			// 2) LETTERBOX viewport/scissor for everything else:
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);

			// draw with the lines pipeline (only once, and only when we have lines)
			if (!lines_vertices.empty()) {
				// we should have allocated a buffer in the upload path
				assert(workspace.lines_vertices.handle != VK_NULL_HANDLE);

				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);

				// use lines_vertices (offset 0) as vertex buffer binding 0:
				{
					VkBuffer vb = workspace.lines_vertices.handle;
					VkDeviceSize off = 0;
					vkCmdBindVertexBuffers(workspace.command_buffer, 0, 1, &vb, &off);
				}

				// bind Camera descriptor set:
				{
					VkDescriptorSet set0 = workspace.Camera_descriptors;
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						lines_pipeline.layout,
						0,
						1, &set0,
						0, nullptr
					);
				}

				// draw lines vertices:
				vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
			}
		}
		{
			// --- MIRROR PASS ---
			if (!mirror_instance_indices.empty()) {

				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mirror_pipeline.handle);
				vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);
				vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);
				// bind packed vertex buffer
				VkBuffer vb = object_vertices.handle;
				VkDeviceSize off = 0;
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, 1, &vb, &off);

				// set 0 = env cubemap
				// set 1 = transforms SSBO
				{
					VkDescriptorSet sets[] = {
						background_pipeline.set,
						workspace.Transforms_descriptors
					};

					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						mirror_pipeline.layout,
						0,
						2,
						sets,
						0,
						nullptr
					);
					 
				}

				// push constants (once per pass is fine — camera is same)
				{
					Tutorial::MirrorPipeline::Push push{};

					mat4 world_from_view = mat4_inverse(this->VIEW_FROM_WORLD);
					push.camera_ws = S72::vec3{
						.x = world_from_view[12],
						.y = world_from_view[13],
						.z = world_from_view[14],
					};

					push.exposure = rtg.configuration.exposure;
					push.tone_op = (rtg.configuration.tone_map == "reinhard") ? 1 : 0;

					vkCmdPushConstants(
						workspace.command_buffer,
						mirror_pipeline.layout,
						VK_SHADER_STAGE_FRAGMENT_BIT,
						0,
						sizeof(push),
						&push
					);
				}

				// draw every mirror instance
				for (uint32_t idx : mirror_instance_indices) {
					auto const& mv = object_instances[idx].vertices;
					vkCmdDraw(
						workspace.command_buffer,
						mv.count,
						1,
						mv.first,
						idx
					);
				}
			}
}

		//objects pass
		{ // --- OBJECTS PASS ---
			 
				if (!object_instances.empty()) {
					 
					for (size_t li = 0; li < 1; ++li) {
						mat4 m = (!shadow_spot_lights.empty())
							? make_spot_light_matrix(*shadow_spot_lights[0])
							: mat4{ 1.0f, 0.0f, 0.0f, 0.0f,
								   0.0f, 1.0f, 0.0f, 0.0f,
								   0.0f, 0.0f, 1.0f, 0.0f,
								   0.0f, 0.0f, 0.0f, 1.0f };
						VkBuffer vb = object_vertices.handle;
						VkDeviceSize off = 0;
						vkCmdBindVertexBuffers(workspace.command_buffer, 0, 1, &vb, &off);

						vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects_pipeline.handle);
						vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);
						vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);

						for (uint32_t i = 0; i < object_instances.size(); ++i) {
							ObjectInstance const& inst = object_instances[i];

							// skip non-lambert objects
							if (inst.material && (std::holds_alternative<S72::Material::PBR>(inst.material->brdf) ||
								std::holds_alternative<S72::Material::Mirror>(inst.material->brdf))) {
								continue;
							}

							uint32_t set2_idx = inst.texture;
							if (use_s72_scene && inst.material) {
								auto it = material_name_to_set2.find(inst.material->name);
								if (it != material_name_to_set2.end()) set2_idx = it->second;
							}

							std::array<VkDescriptorSet, 6> sets = {
								workspace.World_descriptors,                  // set 0
								workspace.Transforms_descriptors,             // set 1
								material_descriptors_lam[set2_idx],           // set 2
								env_lambertian_descriptors,                   // set 3
								workspace.Lights_descriptors,                 // set 4
								workspace.Shadow_descriptors                  // set 5                                                                                                                                       
							};

							vkCmdBindDescriptorSets(
								workspace.command_buffer,
								VK_PIPELINE_BIND_POINT_GRAPHICS,
								objects_pipeline.layout,
								0,
								uint32_t(sets.size()),
								sets.data(),
								0,
								nullptr
							);

							ObjectsPipeline::Push push{};
						 
							 

							 
							push.LIGHT_CLIP_FROM_WORLD = m;
							push.SHADOW_LIGHT_INDEX = -1;

							vkCmdPushConstants(
								workspace.command_buffer,
								objects_pipeline.layout,
								VK_SHADER_STAGE_FRAGMENT_BIT,
								0,
								sizeof(push),
								&push
							);

							vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, i);
						}
					
						//if (false) {
						// --- Inside Tutorial::render ---
						{ // --- PASS B: PBR   ---
							vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_pipeline.handle);
							vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);
							vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);
							// Bind PBR Global Sets (0: World, 1: Transforms, 3: PBR Env)
							VkDescriptorSet pbr_globals[] = {
			workspace.World_descriptors,
			workspace.Transforms_descriptors,
			workspace.PBR_Env_descriptors
							};

							

							// 1. CALCULATE VIEW INVERSE ONCE PER PASS
							mat4 view_to_world = mat4_inverse_rigid(this->VIEW_FROM_WORLD);
							S72::vec3 cam_pos{ view_to_world[12], view_to_world[13], view_to_world[14] };

							for (uint32_t idx : pbr_instance_indices) { // Use the NEW specialized list
								ObjectInstance const& inst = object_instances[idx];
								std::array<VkDescriptorSet, 6> sets = {
   workspace.World_descriptors,                  // set 0
   workspace.Transforms_descriptors,             // set 1
   material_descriptors_pbr[inst.texture],       // set 2
   workspace.PBR_Env_descriptors,                // set 3
   workspace.Lights_descriptors,                 // set 4
   workspace.Shadow_descriptors    // set 5
								};

								vkCmdBindDescriptorSets(
									workspace.command_buffer,
									VK_PIPELINE_BIND_POINT_GRAPHICS,
									pbr_pipeline.layout,
									0,
									uint32_t(sets.size()),
									sets.data(),
									0,
									nullptr
								);
								 

								PBRPipeline::Push push{};
								push.CLIP_FROM_LOCAL = inst.transform.CLIP_FROM_LOCAL;
								push.WORLD_FROM_LOCAL = inst.transform.WORLD_FROM_LOCAL;
								//push.LIGHT_CLIP_FROM_WORLD = inst.transform.LIGHT_CLIP_FROM_WORLD;
								if (!shadow_spot_lights.empty()) {
									push.LIGHT_CLIP_FROM_WORLD = m;
								}
								else {
									push.LIGHT_CLIP_FROM_WORLD = mat4{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
									};
								}

								push.camera_ws = cam_pos;
								push.exposure = rtg.configuration.exposure;
								push.tone_op = (rtg.configuration.tone_map == "reinhard") ? 1 : 0;
								push.SHADOW_LIGHT_INDEX = -1;
								vkCmdPushConstants(workspace.command_buffer, pbr_pipeline.layout,
									VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
									0, sizeof(push), &push);

								vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, idx);
							}
						}
						//}
					}
			}
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}

	//end recording:
	VK(vkEndCommandBuffer(workspace.command_buffer));



	{ //submit `workspace.command buffer` for the GPU to run:

		std::array< VkSemaphore, 1 > wait_semaphores{
			render_params.image_available
		};
		std::array< VkPipelineStageFlags, 1 > wait_stages{
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		};
		static_assert(wait_semaphores.size() == wait_stages.size(), "every semaphore needs a stage");

		std::array< VkSemaphore, 1 > signal_semaphores{
			render_params.image_done
		};
		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = uint32_t(wait_semaphores.size()),
			.pWaitSemaphores = wait_semaphores.data(),
			.pWaitDstStageMask = wait_stages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &workspace.command_buffer,
			.signalSemaphoreCount = uint32_t(signal_semaphores.size()),
			.pSignalSemaphores = signal_semaphores.data(),
		};

		VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, render_params.workspace_available));
	}

	//log one sample per rendered frame
	//frame_logger.tick(double frame_ms);
	//if (rtg.configuration.headless) {
		//frame_logger.tick(double frame_ms);
	//}
}

static mat4 mat4_identity() {
	return mat4{
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
}

static mat4 mat4_translate(float x, float y, float z) {
	return mat4{
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		x,y,z,1
	};
}

static mat4 mat4_scale(float x, float y, float z) {
	return mat4{
		x,0,0,0,
		0,y,0,0,
		0,0,z,0,
		0,0,0,1
	};
}

// quaternion (x,y,z,w) to rotation matrix:
static mat4 mat4_from_quat(float x, float y, float z, float w) {
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, xz = x * z, yz = y * z;
	float wx = w * x, wy = w * y, wz = w * z;

	// column-major mat4 matching your existing usage (translation in last row)
	return mat4{
		1.0f - 2.0f * (yy + zz),  2.0f * (xy + wz),        2.0f * (xz - wy),        0.0f,
		2.0f * (xy - wz),        1.0f - 2.0f * (xx + zz),  2.0f * (yz + wx),        0.0f,
		2.0f * (xz + wy),        2.0f * (yz - wx),        1.0f - 2.0f * (xx + yy),  0.0f,
		0.0f,                  0.0f,                  0.0f,                  1.0f
	};
}

static mat4 mat4_mul(mat4 const& A, mat4 const& B) {
	mat4 R{};
	for (int c = 0; c < 4; ++c) {
		for (int r = 0; r < 4; ++r) {
			R[c * 4 + r] =
				A[0 * 4 + r] * B[c * 4 + 0] +
				A[1 * 4 + r] * B[c * 4 + 1] +
				A[2 * 4 + r] * B[c * 4 + 2] +
				A[3 * 4 + r] * B[c * 4 + 3];
		}
	}
	return R;
}

 

static mat4 mat4_inverse(mat4 const& m) {
	// general 4x4 inverse (column-major)
	float inv[16];

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (std::fabs(det) < 1e-9f) {
		// not invertible; return identity as a safe fallback
		return mat4_identity();
	}

	float invDet = 1.0f / det;

	mat4 out{};
	for (int i = 0; i < 16; i++) out[i] = inv[i] * invDet;
	return out;
}

static void collect_scene_cameras(S72 const& scene, std::vector<S72::Node const*>& out) {
	out.clear();

	std::function<void(S72::Node const&)> walk;
	walk = [&](S72::Node const& n) {
		if (n.camera) out.emplace_back(&n);
		for (S72::Node* child : n.children) {
			if (child) walk(*child);
		}
		};

	for (S72::Node* root : scene.scene.roots) {
		if (root) walk(*root);
	}
}



struct Vec4 { float x, y, z, w; };

static Vec4 mul(mat4 const& M, float x, float y, float z, float w) {
	// mat4 is column-major, and multiply like CLIP_FROM_WORLD * WORLD_FROM_LOCAL
	 
	Vec4 r;
	r.x = M[0] * x + M[4] * y + M[8] * z + M[12] * w;
	r.y = M[1] * x + M[5] * y + M[9] * z + M[13] * w;
	r.z = M[2] * x + M[6] * y + M[10] * z + M[14] * w;
	r.w = M[3] * x + M[7] * y + M[11] * z + M[15] * w;
	return r;
}

static bool aabb_is_culled_in_clip( //https://ktstephano.github.io/rendering/stratusgfx/aabbs helped a lot
	//extract the 8 transformed corners of the box using min/max
	mat4 const& CLIP_FROM_LOCAL,
	float minx, float miny, float minz,
	float maxx, float maxy, float maxz
) {
	// 8 corners in local space:
	float cx[8] = { minx, maxx, minx, maxx, minx, maxx, minx, maxx };
	float cy[8] = { miny, miny, maxy, maxy, miny, miny, maxy, maxy };
	float cz[8] = { minz, minz, minz, minz, maxz, maxz, maxz, maxz };

	// For each plane, if all corners are outside -> cull:
	int out;

	// left: x < -w
	out = 0;
	for (int i = 0;i < 8;i++) {
		Vec4 p = mul(CLIP_FROM_LOCAL, cx[i], cy[i], cz[i], 1.0f);
		if (p.x < -p.w) out++;
	}
	if (out == 8) return true;

	// right: x > w
	out = 0;
	for (int i = 0;i < 8;i++) {
		Vec4 p = mul(CLIP_FROM_LOCAL, cx[i], cy[i], cz[i], 1.0f);
		if (p.x > p.w) out++;
	}
	if (out == 8) return true;

	// bottom: y < -w
	out = 0;
	for (int i = 0;i < 8;i++) {
		Vec4 p = mul(CLIP_FROM_LOCAL, cx[i], cy[i], cz[i], 1.0f);
		if (p.y < -p.w) out++;
	}
	if (out == 8) return true;

	// top: y > w
	out = 0;
	for (int i = 0;i < 8;i++) {
		Vec4 p = mul(CLIP_FROM_LOCAL, cx[i], cy[i], cz[i], 1.0f);
		if (p.y > p.w) out++;
	}
	if (out == 8) return true;

	// near (Vulkan): z < 0
	out = 0;
	for (int i = 0;i < 8;i++) {
		Vec4 p = mul(CLIP_FROM_LOCAL, cx[i], cy[i], cz[i], 1.0f);
		if (p.z < 0.0f) out++;
	}
	if (out == 8) return true;

	// far: z > w
	out = 0;
	for (int i = 0;i < 8;i++) {
		Vec4 p = mul(CLIP_FROM_LOCAL, cx[i], cy[i], cz[i], 1.0f);
		if (p.z > p.w) out++;
	}
	if (out == 8) return true;

	return false; // not fully outside any plane -> keep
}



static void add_aabb_lines( //heavily based on https://ktstephano.github.io/rendering/stratusgfx/aabbs
	std::vector<PosColVertex>& out,
	mat4 const& WORLD_FROM_LOCAL,
	S72::vec3 const& local_min,
	S72::vec3 const& local_max,
	uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff
) {
	auto xform = [&](float x, float y, float z) -> S72::vec3 {
		Vec4 p = mul(WORLD_FROM_LOCAL, x, y, z, 1.0f);
		return S72::vec3{ p.x, p.y, p.z };
		};

	//create the 8 corners
	S72::vec3 c[8];
	c[0] = xform(local_min.x, local_min.y, local_min.z);
	c[1] = xform(local_max.x, local_min.y, local_min.z);
	c[2] = xform(local_min.x, local_max.y, local_min.z);
	c[3] = xform(local_max.x, local_max.y, local_min.z);
	c[4] = xform(local_min.x, local_min.y, local_max.z);
	c[5] = xform(local_max.x, local_min.y, local_max.z);
	c[6] = xform(local_min.x, local_max.y, local_max.z);
	c[7] = xform(local_max.x, local_max.y, local_max.z);

	auto push_seg = [&](int i0, int i1) { //drawing help
		out.emplace_back(PosColVertex{
			.Position{.x = c[i0].x, .y = c[i0].y, .z = c[i0].z },
			.Color{.r = r, .g = g, .b = b, .a = a }
			});
		out.emplace_back(PosColVertex{
			.Position{.x = c[i1].x, .y = c[i1].y, .z = c[i1].z },
			.Color{.r = r, .g = g, .b = b, .a = a }
			});
		};

	// connect the corners to form 12 lines
	push_seg(0, 1); push_seg(1, 3); push_seg(3, 2); push_seg(2, 0);
	push_seg(4, 5); push_seg(5, 7); push_seg(7, 6); push_seg(6, 4);
	push_seg(0, 4); push_seg(1, 5); push_seg(2, 6); push_seg(3, 7);
}

static S72::vec3 clip_to_world(mat4 const& WORLD_FROM_CLIP, float x, float y, float z) {
	Vec4 p = mul(WORLD_FROM_CLIP, x, y, z, 1.0f);
	float iw = (p.w != 0.0f) ? (1.0f / p.w) : 0.0f;
	return S72::vec3{ p.x * iw, p.y * iw, p.z * iw };
}

static void add_line(
	std::vector<PosColVertex>& out,
	S72::vec3 a, S72::vec3 b,
	uint8_t r, uint8_t g, uint8_t bb, uint8_t a8 = 0xff
) {
	out.emplace_back(PosColVertex{
		.Position{.x = a.x, .y = a.y, .z = a.z},
		.Color{.r = r, .g = g, .b = bb, .a = a8}
		});
	out.emplace_back(PosColVertex{
		.Position{.x = b.x, .y = b.y, .z = b.z},
		.Color{.r = r, .g = g, .b = bb, .a = a8}
		});
}

static void add_frustum_lines_from_clip(
	std::vector<PosColVertex>& out,
	mat4 const& CLIP_FROM_WORLD,
	uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff
) {
	mat4 WORLD_FROM_CLIP = mat4_inverse(CLIP_FROM_WORLD);

	// Vulkan NDC: x,y in [-1,1], z in [0,1]
	S72::vec3 n0 = clip_to_world(WORLD_FROM_CLIP, -1, -1, 0);
	S72::vec3 n1 = clip_to_world(WORLD_FROM_CLIP, 1, -1, 0);
	S72::vec3 n2 = clip_to_world(WORLD_FROM_CLIP, -1, 1, 0);
	S72::vec3 n3 = clip_to_world(WORLD_FROM_CLIP, 1, 1, 0);

	S72::vec3 f0 = clip_to_world(WORLD_FROM_CLIP, -1, -1, 1);
	S72::vec3 f1 = clip_to_world(WORLD_FROM_CLIP, 1, -1, 1);
	S72::vec3 f2 = clip_to_world(WORLD_FROM_CLIP, -1, 1, 1);
	S72::vec3 f3 = clip_to_world(WORLD_FROM_CLIP, 1, 1, 1);

	// near rect
	add_line(out, n0, n1, r, g, b, a);
	add_line(out, n1, n3, r, g, b, a);
	add_line(out, n3, n2, r, g, b, a);
	add_line(out, n2, n0, r, g, b, a);

	// far rect
	add_line(out, f0, f1, r, g, b, a);
	add_line(out, f1, f3, r, g, b, a);
	add_line(out, f3, f2, r, g, b, a);
	add_line(out, f2, f0, r, g, b, a);

	// sides
	add_line(out, n0, f0, r, g, b, a);
	add_line(out, n1, f1, r, g, b, a);
	add_line(out, n2, f2, r, g, b, a);
	add_line(out, n3, f3, r, g, b, a);
}

static void apply_drivers(S72& s72, float t) {
	for (auto& drv : s72.drivers) {
		// 1) find keyframe interval
		auto const& times = drv.times;
		auto const& vals = drv.values;
		if (times.empty()) continue;

		size_t i = 0;
		if (t <= times.front()) {
			i = 0;
		}
		else if (t >= times.back()) {
			i = times.size() - 1;
		}
		else {
			// find i such that times[i] <= t < times[i+1]
			while (i + 1 < times.size() && !(times[i] <= t && t < times[i + 1])) ++i;
		}

		// helper lambdas to read key values:
		auto read_vec3 = [&](size_t key)->S72::vec3 {
			size_t base = key * 3;
			return S72::vec3{ vals[base + 0], vals[base + 1], vals[base + 2] };
			};
		auto read_quat = [&](size_t key)->S72::quat {
			size_t base = key * 4;
			return S72::quat{ vals[base + 0], vals[base + 1], vals[base + 2], vals[base + 3] };
			};

		// 2) apply (start with STEP ONLY to get moving fast)
		if (drv.channel == S72::Driver::Channel::translation) {
			drv.node.translation = read_vec3(i);
		}
		else if (drv.channel == S72::Driver::Channel::scale) {
			drv.node.scale = read_vec3(i);
		}
		else { // rotation
			drv.node.rotation = read_quat(i);
		}
	}
}


void Tutorial::compute_letterbox(float target_aspect) {
	//target_aspect == 0 => full screen
	float W = float(rtg.swapchain_extent.width);
	float H = float(rtg.swapchain_extent.height);

	//default full screen:
	draw_viewport = VkViewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = W,
		.height = H,
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	draw_scissor = VkRect2D{
		.offset = {0, 0},
		.extent = rtg.swapchain_extent,
	};

	if (target_aspect <= 0.0f) return;

	float win_aspect = W / H;

	//If camera is wider than window => letterbox (bars top/bottom)
	//If camera is taller than window => pillarbox (bars left/right)
	if (target_aspect > win_aspect) {
		//fit width
		float newH = W / target_aspect;
		float y0 = (H - newH) * 0.5f;

		draw_viewport.y = y0;
		draw_viewport.height = newH;

		draw_scissor.offset.y = int32_t(std::round(y0));
		draw_scissor.extent.height = uint32_t(std::round(newH));
	}
	else {
		//fit height
		float newW = H * target_aspect;
		float x0 = (W - newW) * 0.5f;

		draw_viewport.x = x0;
		draw_viewport.width = newW;

		draw_scissor.offset.x = int32_t(std::round(x0));
		draw_scissor.extent.width = uint32_t(std::round(newW));
	}
}



void Tutorial::update(float dt) {

	

	if (forced_camera.has_value()) {
		camera_mode = forced_camera.value();
	}


	//time += dt;
	time = std::fmod(time + dt, 60.0f);

	//advance and apply animation driver
	 

	if (use_s72_scene) {
		if (!anim_started) {
			anim_time = 0.0f;        // first rendered frame at 0
			anim_started = true;
		}
		else if (!anim_paused) {
			anim_time += dt;         // advance after first frame
		}
		apply_drivers(scene, anim_time);

		loaded_lights.clear();

		mat4 identity = mat4_identity();
		for (auto const* root : scene.scene.roots) {
			collect_loaded_lights_from_node(root, identity, loaded_lights);
		}
	}

	auto local_from_node = [&](S72::Node const& n) -> mat4 {
		mat4 T = mat4_translate(n.translation.x, n.translation.y, n.translation.z);
		mat4 R = mat4_from_quat(n.rotation.x, n.rotation.y, n.rotation.z, n.rotation.w);
		mat4 S = mat4_scale(n.scale.x, n.scale.y, n.scale.z);
		return mat4_mul(T, mat4_mul(R, S)); // TRS
		};

	auto world_from_node = [&](S72::Node const& n) -> mat4 {
		// compute by walking parents using the node's .parent pointer if it exists.
		// If Node doesn't have parent, we compute via recursion from roots (fallback below).
		mat4 W = mat4_identity();

		// Fallback path: recompute from roots every time  
		//   do a quick DFS until we hit this node.
		bool found = false;
		std::function<void(S72::Node const&, mat4 const&)> find;
		find = [&](S72::Node const& cur, mat4 const& parentW) {
			if (found) return;
			mat4 Wcur = mat4_mul(parentW, local_from_node(cur));
			if (&cur == &n) {
				W = Wcur;
				found = true;
				return;
			}
			for (S72::Node* ch : cur.children) if (ch) find(*ch, Wcur);
			};

		for (S72::Node* root : scene.scene.roots) if (root) find(*root, mat4_identity());
		return W;
		};

	 

	auto set_from_orbit = [&](OrbitCamera const& cam) {
		CLIP_FROM_VIEW = perspective(
			cam.fov,
			rtg.swapchain_extent.width / float(rtg.swapchain_extent.height),
			cam.near, cam.far
		);
		VIEW_FROM_WORLD = orbit(
			cam.target_x, cam.target_y, cam.target_z,
			cam.azimuth, cam.elevation, cam.radius
		);

		CLIP_FROM_WORLD = CLIP_FROM_VIEW * VIEW_FROM_WORLD;
		
		};

	if (camera_mode == CameraMode::Scene) {
		if (scene_camera_nodes.empty()) {
			set_from_orbit(free_camera);
			CLIP_FROM_CULL = CLIP_FROM_WORLD;

			scene_cam_aspect = 0.0f;
			compute_letterbox(0.0f);
		}
		else {
			active_scene_camera = std::min(active_scene_camera, uint32_t(scene_camera_nodes.size() - 1));
			S72::Node const& cam_node = *scene_camera_nodes[active_scene_camera];

			mat4 WORLD_FROM_CAMERA = world_from_node(cam_node);
			mat4 CAMERA_FROM_WORLD = mat4_inverse_rigid(WORLD_FROM_CAMERA); // this is VIEW_FROM_WORLD

			float vfov = 60.0f * float(M_PI) / 180.0f;
			float near_ = 0.1f;
			float far_ = 1000.0f;
			scene_cam_aspect = rtg.swapchain_extent.width / float(rtg.swapchain_extent.height);

			if (cam_node.camera) {
				if (auto const* persp = std::get_if<S72::Camera::Perspective>(&cam_node.camera->projection)) {
					vfov = persp->vfov;
					near_ = persp->near;
					far_ = persp->far;
					if (persp->aspect > 0.0f) scene_cam_aspect = persp->aspect;
				}
			}
			if (!std::isfinite(far_) || far_ <= near_) far_ = 1000.0f;

			compute_letterbox(scene_cam_aspect);

			VIEW_FROM_WORLD = CAMERA_FROM_WORLD;
			CLIP_FROM_VIEW = perspective(vfov, scene_cam_aspect, near_, far_);

			CLIP_FROM_WORLD = CLIP_FROM_VIEW * VIEW_FROM_WORLD;
			CLIP_FROM_CULL = CLIP_FROM_WORLD;
		}
	}
	else if (camera_mode == CameraMode::User) {
		set_from_orbit(free_camera);
		CLIP_FROM_CULL = CLIP_FROM_WORLD;

		scene_cam_aspect = 0.0f;
		compute_letterbox(0.0f);
	}
	else if (camera_mode == CameraMode::Debug) {
		set_from_orbit(debug_camera);

		if (debug_cull_locked) CLIP_FROM_CULL = debug_locked_CLIP_FROM_CULL;
		else CLIP_FROM_CULL = CLIP_FROM_WORLD;

		scene_cam_aspect = 0.0f;
		compute_letterbox(0.0f);
	}

	 




	{ //static sun and sky:
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = 1.0f;

		world.SKY_ENERGY.r = 0.1f;
		world.SKY_ENERGY.g = 0.1f;
		world.SKY_ENERGY.b = 0.2f;

		world.SUN_DIRECTION.x = 6.0f / 23.0f;
		world.SUN_DIRECTION.y = 13.0f / 23.0f;
		world.SUN_DIRECTION.z = 18.0f / 23.0f;

		world.SUN_ENERGY.r = 1.0f;
		world.SUN_ENERGY.g = 1.0f;
		world.SUN_ENERGY.b = 0.9f;
	}



	//
	//lines_vertices.reserve(4);
	/* { //make some crossing lines at different depths:
		lines_vertices.clear();


		const int N = 60;
		const float size = 1.0f;
		const float step = (2.0f * size) / float(N);
		size_t count = 4 * (N + 1) * N;
		lines_vertices.reserve(count);


		auto rippleY = [&](float x, float z) -> float {
			float d = std::sqrt(x * x + z * z);
			float y = std::sin(float(M_PI) * (4.0f * d - time));
			y /= (1.0f + 10.0f * d);
			return y;
			};

		// rows (z fixed, x changes)
		for (int zi = 0; zi <= N; ++zi) {
			float z = -size + zi * step;
			for (int xi = 0; xi < N; ++xi) {
				float x0 = -size + xi * step;
				float x1 = x0 + step;

				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x0, .y = rippleY(x0, z), .z = z },
					.Color{.r = 0xff, .g = 0xff, .b = 0x00, .a = 0xff }
					});
				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x1, .y = rippleY(x1, z), .z = z },
					.Color{.r = 0xff, .g = 0xff, .b = 0x00, .a = 0xff }
					});
			}
		}

		// columns (x fixed, z changes)
		for (int xi = 0; xi <= N; ++xi) {
			float x = -size + xi * step;
			for (int zi = 0; zi < N; ++zi) {
				float z0 = -size + zi * step;
				float z1 = z0 + step;

				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x, .y = rippleY(x, z0), .z = z0 },
					.Color{.r = 0x44, .g = 0x00, .b = 0xff, .a = 0xff }
					});
				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x, .y = rippleY(x, z1), .z = z1 },
					.Color{.r = 0x44, .g = 0x00, .b = 0xff, .a = 0xff }
					});
			}
		}

		assert(lines_vertices.size() == count);
	}*/
	lines_vertices.clear();
	lines_vertices.reserve(100000); // 

	if (camera_mode == CameraMode::Debug) {
		// Draw the culling frustum (from the locked/previous camera)
		add_frustum_lines_from_clip(lines_vertices, CLIP_FROM_CULL, 0x20, 0x80, 0xff, 0xff);
	}

	 
	 

	{ //make some objects:
		object_instances.clear();
		mirror_instance_indices.clear();
		pbr_instance_indices.clear();

		if (!scene_file.empty()) {
			// build instances from S72 scene nodes/meshes
			auto local_from_node2 = [&](S72::Node const& n) -> mat4 {
				mat4 T = mat4_translate(n.translation.x, n.translation.y, n.translation.z);
				mat4 R = mat4_from_quat(n.rotation.x, n.rotation.y, n.rotation.z, n.rotation.w);
				mat4 S = mat4_scale(n.scale.x, n.scale.y, n.scale.z);
				return mat4_mul(T, mat4_mul(R, S));
				};


			std::function<void(S72::Node const&, mat4 const&)> emit_node;
			emit_node = [&](S72::Node const& n, mat4 const& parent_world) {
				mat4 WORLD_FROM_LOCAL = mat4_mul(parent_world, local_from_node2(n));
				

				if (n.mesh) {
					auto it = s72_mesh_to_range.find(n.mesh);
					if (it != s72_mesh_to_range.end() && it->second.count > 0) {

						ObjectVertices const& vr = it->second;

						// build clip-from-local for culling  CLIP_FROM_CULL
						mat4 CLIP_FROM_LOCAL_CULL = mat4_mul(CLIP_FROM_CULL, WORLD_FROM_LOCAL);

						bool would_cull = aabb_is_culled_in_clip(
							CLIP_FROM_LOCAL_CULL,
							vr.local_min.x, vr.local_min.y, vr.local_min.z,
							vr.local_max.x, vr.local_max.y, vr.local_max.z
						);

						bool culled = (enable_culling && would_cull);

						if (camera_mode == CameraMode::Debug) {
							// color scheme:
							// - red: would be culled AND culling enabled (actually skipped)
							// - yellow: would be culled BUT culling disabled (still drawn)
							// - green: inside frustum
							if (would_cull && enable_culling) {
								add_aabb_lines(lines_vertices, WORLD_FROM_LOCAL, vr.local_min, vr.local_max, 0xff, 0x20, 0x20);
							}
							else if (would_cull && !enable_culling) {
								add_aabb_lines(lines_vertices, WORLD_FROM_LOCAL, vr.local_min, vr.local_max, 0xff, 0xff, 0x20);
							}
							else {
								add_aabb_lines(lines_vertices, WORLD_FROM_LOCAL, vr.local_min, vr.local_max, 0x20, 0xff, 0x20);
							}
						}
						// pick set2 index from material name (fallback to 0)
						uint32_t set2_idx = 0;
						if (n.mesh->material) {
							auto itM = material_name_to_set2.find(n.mesh->material->name);
							if (itM != material_name_to_set2.end()) set2_idx = itM->second;
						}

						if ((!culled)) {
							bool is_mirror = false;
							bool is_pbr = false; // New flag

							if (n.mesh && n.mesh->material) {
								// Categorize the material type once at startup/update
								is_mirror = std::holds_alternative<S72::Material::Mirror>(n.mesh->material->brdf);
								is_pbr = std::holds_alternative<S72::Material::PBR>(n.mesh->material->brdf);
							}

							object_instances.emplace_back(ObjectInstance{
								.vertices = vr,
								.transform{
									.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
									.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
									.WORLD_FROM_LOCAL_NORMAL = mat4_transpose(mat4_inverse(WORLD_FROM_LOCAL)),
								},
								.texture = set2_idx,
								.material = n.mesh->material,
								});

							// Store the index in the appropriate specialized list
							if (is_mirror) {
								mirror_instance_indices.emplace_back(uint32_t(object_instances.size() - 1));
							}
							else if (is_pbr) {
								pbr_instance_indices.emplace_back(uint32_t(object_instances.size() - 1));
							}
						}
					}
				}


				for (S72::Node* child : n.children) {
					if (child) emit_node(*child, WORLD_FROM_LOCAL);
				}
				};

			mat4 I = mat4_identity();
			for (S72::Node* root : scene.scene.roots) {
				if (root) emit_node(*root, I);
			}


			

		}
		else {
			//fallback: old hardcoded objects (optional)
			// (you can keep this branch if you want a non-scene demo mode)
			{ //plane translated +x by one unit:
				mat4 WORLD_FROM_LOCAL{
					1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					1.0f, 0.0f, 0.0f, 1.0f,
				};

				object_instances.emplace_back(ObjectInstance{
					.vertices = plane_vertices,
					.transform{
						.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
					},
					.texture = 1,
					});
			}
			{ //torus translated -x by one unit and rotated CCW around +y:
				float ang = time / 60.0f * 2.0f * float(M_PI) * 10.0f;
				float ca = std::cos(ang);
				float sa = std::sin(ang);
				mat4 WORLD_FROM_LOCAL{
					  ca, 0.0f,  -sa, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					  sa, 0.0f,   ca, 0.0f,
					-1.0f,0.0f, 0.0f, 1.0f,
				};

				object_instances.emplace_back(ObjectInstance{
					.vertices = torus_vertices,
					.transform{
						.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
					},
					});
			}

			{ //chen parts near origin
// if she’s huge/small, scale here later
				float s = 0.05f;

				mat4 WORLD_FROM_LOCAL{
					s,    0.0f, 0.0f, 0.0f,
					0.0f, s,    0.0f, 0.0f,
					0.0f, 0.0f, s,    0.0f,
					0.0f, -0.5f, 0.0f, 1.0f,
				};



				auto add_part = [&](ObjectVertices vr, uint32_t tex) {
					object_instances.emplace_back(ObjectInstance{
						.vertices = vr,
						.transform{
							.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
							.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
							.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
						},
						.texture = tex,
						});
					};

				add_part(chen_body_vertices, tex_body);
				add_part(chen_clothes_vertices, tex_clothes);
				add_part(chen_hairs_vertices, tex_hair);
				add_part(chen_face_vertices, tex_face);
				add_part(chen_iris_vertices, tex_iris);

				// optional:
				add_part(chen_sword_vertices, tex_sword);

			}


		
		}
	}
}
void Tutorial::on_input(InputEvent const &evt) {
	//if there is a current action, it gets input priority:
	if (action) {
		action(evt);
		return;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_R) {
		anim_time = 0.0f;
		anim_started = true;          // keep first-frame rule stable
		apply_drivers(scene, anim_time);
		std::cout << "[A1-move] restart anim_time=0\n";
		return;
	}

	// animation controls 
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_SPACE) {
		anim_paused = !anim_paused;
		std::cout << "[A1-move] anim " << (anim_paused ? "PAUSED\n" : "PLAYING\n");
		return;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_N) {
		anim_time += 1.0f / 30.0f; // step one frame at 30fps
		apply_drivers(scene, anim_time);
		std::cout << "[A1-move] step anim_time=" << anim_time << "\n";
		return;
	}


	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_V) {
		enable_culling = !enable_culling;
		std::cout << "[A1-cull] culling: " << (enable_culling ? "ON\n" : "OFF\n");
		return;
	}

	//general controls:
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB) {
		CameraMode prev = camera_mode;
		camera_mode = CameraMode((int(camera_mode) + 1) % 3);

		// if we are ENTERING debug, lock cull camera to whatever it was
		if (camera_mode == CameraMode::Debug && prev != CameraMode::Debug) {
			debug_cull_locked = true;
			debug_locked_CLIP_FROM_CULL = CLIP_FROM_CULL; // from last update()
		}

		// if we are LEAVING debug, unlock
		if (prev == CameraMode::Debug && camera_mode != CameraMode::Debug) {
			debug_cull_locked = false;
		}
		if (camera_mode == CameraMode::Scene) std::cout << "[A1-show] camera mode: Scene\n";
		if (camera_mode == CameraMode::User)  std::cout << "[A1-show] camera mode: User\n";
		if (camera_mode == CameraMode::Debug) std::cout << "[A1-show] camera mode: Debug\n";

		return;
	}


	// cycle scene cameras (only when in scene mode):
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_C) {
		if (camera_mode == CameraMode::Scene && !scene_camera_nodes.empty()) {
			active_scene_camera = (active_scene_camera + 1) % uint32_t(scene_camera_nodes.size());
			std::cout << "[A1-show] active scene camera: " << active_scene_camera << " / " << scene_camera_nodes.size() << "\n";
		}
		return;
	}

	
	// user/debug orbit controls
	if (camera_mode == CameraMode::User || camera_mode == CameraMode::Debug) {
		//free camera controls
		OrbitCamera& cam = (camera_mode == CameraMode::Debug ? debug_camera : free_camera);

		//This camera move is a "dolly" not a "zoom" because 
		//we're moving the camera's position, not changing its field of view.

		if (evt.type == InputEvent::MouseWheel) {
			//change distance by 10% every scroll click:
			cam.radius *= std::exp(std::log(1.1f) * -evt.wheel.y);
			//make sure camera isn't too close or too far from target:
			cam.radius = std::max(cam.radius, 0.5f * cam.near);
			cam.radius = std::min(cam.radius, 2.0f * cam.far);
			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT
			&& (evt.button.mods & GLFW_MOD_SHIFT)) {
			//start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = cam;

			action = [this, init_x, init_y, init_camera, &cam](InputEvent const& evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//image height at plane of target point:
					float height = 2.0f * std::tan(init_camera.fov * 0.5f) * init_camera.radius;

					//motion, therefore, at target point:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy = (evt.motion.y - init_y) / rtg.swapchain_extent.height * height; //note: negated because glfw uses y-down

					//compute camera transform to extract right (first row) and up (second row):
					mat4 camera_from_world = orbit(
						init_camera.target_x, init_camera.target_y, init_camera.target_z,
						init_camera.azimuth, init_camera.elevation, init_camera.radius
					);

					//move the desired distance:
					cam.target_x = init_camera.target_x - dx * camera_from_world[0] - dy * camera_from_world[1];
					cam.target_y = init_camera.target_y - dx * camera_from_world[4] - dy * camera_from_world[5];
					cam.target_z = init_camera.target_z - dx * camera_from_world[8] - dy * camera_from_world[9];

					return;
				}
			};
			return;
		}

		 
	


		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
			//start tumbling (rotate current cam)

			 

			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = cam;

			action = [this, init_x, init_y, init_camera, &cam](InputEvent const& evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height; //glfw y-down

					float speed = float(M_PI);
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f);

					cam.azimuth = init_camera.azimuth - dx * speed * flip_x;
					cam.elevation = init_camera.elevation - dy * speed;

					const float twopi = 2.0f * float(M_PI);
					cam.azimuth -= std::round(cam.azimuth / twopi) * twopi;
					cam.elevation -= std::round(cam.elevation / twopi) * twopi;
					return;
				}
				};

			return;
		}
	}

}
	

