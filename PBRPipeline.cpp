#include "tutorial.hpp"

#include "Helpers.hpp"

#include "VK.hpp"

static uint32_t vert_code[] =
#include "spv/pbr.vert.inl"
;

static uint32_t frag_code[] =
#include "spv/pbr.frag.inl"
;

void Tutorial::PBRPipeline::create(RTG& rtg, VkRenderPass render_pass, uint32_t subpass) {
	VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);
	VkShaderModule frag_module = rtg.helpers.create_shader_module(frag_code);

	{ // set0_World (same as ObjectsPipeline)
		std::array<VkDescriptorSetLayoutBinding, 1> bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
			},
		};

		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_World)); //Shaders will be allowed to access a uniform buffer 
		//at set0 binding0 in fragment stage (define expected output)
	}

	{ // set1_Transforms (same)
		std::array<VkDescriptorSetLayoutBinding, 1> bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
			},
		};

		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_Transforms));
	}

	{ // set2_TEXTURE (albedo + normal + roughness + metalness)
		std::array<VkDescriptorSetLayoutBinding, 4> bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
			},
			VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
			},
			VkDescriptorSetLayoutBinding{
				.binding = 2,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
			},
			VkDescriptorSetLayoutBinding{
				.binding = 3,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
			},
		};


		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set2_TEXTURE));
	}

	{ // set3_EnvPBR: lambertian + ggx + brdf lut
		std::array<VkDescriptorSetLayoutBinding, 3> bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
			VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
			VkDescriptorSetLayoutBinding{
				.binding = 2,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
		};

		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set3_EnvPBR));
	}

	{ // set4_Lights
		VkDescriptorSetLayoutBinding lights_binding{};
		lights_binding.binding = 0;
		lights_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		lights_binding.descriptorCount = 1;
		lights_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		lights_binding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo lights_info{};
		lights_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		lights_info.bindingCount = 1;
		lights_info.pBindings = &lights_binding;

		VK(vkCreateDescriptorSetLayout(rtg.device, &lights_info, nullptr, &set4_Lights));
	}

	{//set5_Shadow
		constexpr uint32_t MAX_SHADOW_SPOT_LIGHTS = 16;
		VkDescriptorSetLayoutBinding shadow_binding{};
		shadow_binding.binding = 0;
		shadow_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		shadow_binding.descriptorCount = MAX_SHADOW_SPOT_LIGHTS;
		shadow_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		shadow_binding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo shadow_info{};
		shadow_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		shadow_info.bindingCount = 1;
		shadow_info.pBindings = &shadow_binding;

		VK(vkCreateDescriptorSetLayout(rtg.device, &shadow_info, nullptr, &set5_Shadow));

	}

	{ // pipeline layout
		std::array<VkDescriptorSetLayout, 6> layouts{
	set0_World, set1_Transforms, set2_TEXTURE, set3_EnvPBR, set4_Lights, set5_Shadow
		};

		// Inside Tutorial::PBRPipeline::create
		VkPushConstantRange push_constant_range{
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(Tutorial::PBRPipeline::Push) // Use the 32-byte PBR push struct
		};

		VkPipelineLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(layouts.size()),
			.pSetLayouts = layouts.data(),
			.pushConstantRangeCount = 1,                 // <--- MUST BE 1
			.pPushConstantRanges = &push_constant_range, // <--- PASS THE RANGE
		};

		VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
	}

	{ // pipeline state (copy/paste from ObjectsPipeline )
		std::array<VkPipelineColorBlendAttachmentState, 1> attachment_states{
			VkPipelineColorBlendAttachmentState{
	.blendEnable = VK_FALSE,
	.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
					  VK_COLOR_COMPONENT_G_BIT |
					  VK_COLOR_COMPONENT_B_BIT |
					  VK_COLOR_COMPONENT_A_BIT,
},
		};

		VkPipelineColorBlendStateCreateInfo color_blend_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.attachmentCount = uint32_t(attachment_states.size()),
			.pAttachments = attachment_states.data(),
			.blendConstants{0.0f, 0.0f, 0.0f, 0.0f},
		};

		VkPipelineDepthStencilStateCreateInfo depth_stencil_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};

		VkPipelineMultisampleStateCreateInfo multisample_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
		};

		VkPipelineRasterizationStateCreateInfo rasterization_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.lineWidth = 1.0f,
		};

		VkPipelineViewportStateCreateInfo viewport_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		VkPipelineInputAssemblyStateCreateInfo input_assembly_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		std::vector<VkDynamicState> dynamic_states{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
		};

		VkPipelineDynamicStateCreateInfo dynamic_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = uint32_t(dynamic_states.size()),
			.pDynamicStates = dynamic_states.data(),
		};

		std::array<VkPipelineShaderStageCreateInfo, 2> stages{
			VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vert_module,
				.pName = "main",
			},
			VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = frag_module,
				.pName = "main",
			},
		};

		// ... (Keep the depth/blend states you already have here) ...

		VkGraphicsPipelineCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = uint32_t(stages.size()),
			.pStages = stages.data(),
			// --- FIX THE VERTEX STRUCT SO TANGENTS LOAD ---
			.pVertexInputState = &PosNorTexVertex::array_input_state,
			.pInputAssemblyState = &input_assembly_state,
			.pViewportState = &viewport_state,
			.pRasterizationState = &rasterization_state,
			.pMultisampleState = &multisample_state,
			.pDepthStencilState = &depth_stencil_state,
			.pColorBlendState = &color_blend_state,
			.pDynamicState = &dynamic_state,
			.layout = layout,
			.renderPass = render_pass,
			.subpass = subpass,
		};

		VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
	}

	vkDestroyShaderModule(rtg.device, frag_module, nullptr); //can delete shader modules after creation because the pipeline has consummed them
	vkDestroyShaderModule(rtg.device, vert_module, nullptr);
}





 

 






//destroying pipeline 
void Tutorial::PBRPipeline::destroy(RTG& rtg) {
	//refsol::BackgroundPipeline_destroy(rtg, &layout, &handle);
	//if (set0_Camera != VK_NULL_HANDLE) { //MAKING DSL WITH A SINGLE BINDING
		//vkDestroyDescriptorSetLayout(rtg.device, set0_Camera, nullptr);
		//set0_Camera = VK_NULL_HANDLE;
	//}

	 
	

	if (set1_Transforms != VK_NULL_HANDLE) { 
		vkDestroyDescriptorSetLayout(rtg.device, set1_Transforms, nullptr);
		set1_Transforms = VK_NULL_HANDLE;
	}
	if (set0_World != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_World, nullptr);
		set0_World = VK_NULL_HANDLE;
	}


	if (set2_TEXTURE != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set2_TEXTURE, nullptr);
		set2_TEXTURE = VK_NULL_HANDLE;
	}

	if (set3_EnvPBR != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set3_EnvPBR, nullptr);
		set3_EnvPBR = VK_NULL_HANDLE;
	}

	if (layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(rtg.device, layout, nullptr);
		layout = VK_NULL_HANDLE;
	}

	if (handle != VK_NULL_HANDLE) {
		vkDestroyPipeline(rtg.device, handle, nullptr);
		handle = VK_NULL_HANDLE;
	}

	if (set4_Lights != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set4_Lights, nullptr);
		set4_Lights = VK_NULL_HANDLE;
	}

	if (set5_Shadow != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set5_Shadow, nullptr);
		set5_Shadow = VK_NULL_HANDLE;
	}
}



 

