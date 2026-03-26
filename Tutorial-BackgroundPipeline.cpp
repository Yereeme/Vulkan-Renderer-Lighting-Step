#include "tutorial.hpp"

#include "Helpers.hpp"

#include "VK.hpp"

static uint32_t vert_code[] =
#include "spv/background.vert.inl"
;

static uint32_t frag_code[] =
#include "spv/background.frag.inl"
;

void Tutorial::BackgroundPipeline::create(RTG& rtg, VkRenderPass render_pass, uint32_t subpass) {
	VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);
	VkShaderModule frag_module = rtg.helpers.create_shader_module(frag_code);

	// ---- descriptor set layout (set 0) ----
	std::array<VkDescriptorSetLayoutBinding, 1> bindings = { VkDescriptorSetLayoutBinding{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,// for a combined image sampler descriptor
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, //we intend to use the combined image sampler descriptor in the fragment shader.
		.pImmutableSamplers = nullptr,
	} };

	VkDescriptorSetLayoutCreateInfo set_layout_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(bindings.size()),
		.pBindings = bindings.data()
	};

	VK(vkCreateDescriptorSetLayout(rtg.device, &set_layout_info, nullptr, &set_layout));


	// ---- pipeline layout (set 0 + push constants) ----
	VkPushConstantRange pcr{
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(Push),
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pcr,
	};

	VK(vkCreatePipelineLayout(rtg.device, &pipeline_layout_info, nullptr, &layout));

	// ---- descriptor pool + allocate set ----
	VkDescriptorPoolSize pool_size{
		.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
	};

	VkDescriptorPoolCreateInfo pool_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	VK(vkCreateDescriptorPool(rtg.device, &pool_info, nullptr, &descriptor_pool));

	VkDescriptorSetAllocateInfo alloc_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &set_layout,
	};

	VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &set));

	// ---- pipeline ----
	VkPipelineShaderStageCreateInfo stages[2] = {
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

	// full-screen triangle: no vertex buffers
	VkPipelineVertexInputStateCreateInfo vertex_input{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = nullptr,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = nullptr,
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	VkPipelineViewportStateCreateInfo viewport_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = nullptr, // dynamic
		.scissorCount = 1,
		.pScissors = nullptr,  // dynamic
	};

	VkPipelineRasterizationStateCreateInfo raster{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo msaa{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// background writes color, no depth
	VkPipelineDepthStencilStateCreateInfo depth{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_FALSE,
		.depthWriteEnable = VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_ALWAYS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState blend_attachment{
		.blendEnable = VK_FALSE,
		.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo blend{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dyn{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dyn_states,
	};



	VkGraphicsPipelineCreateInfo pipe_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &raster,
		.pMultisampleState = &msaa,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blend,
		.pDynamicState = &dyn,
		.layout = layout,
		.renderPass = render_pass,
		.subpass = subpass,
	};

	VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &handle));

	// shader modules can go away after pipeline creation
	vkDestroyShaderModule(rtg.device, frag_module, nullptr);
	vkDestroyShaderModule(rtg.device, vert_module, nullptr);
}

	


	


//destroying pipeline (lazy
void Tutorial::BackgroundPipeline::destroy(RTG& rtg) {
	if (handle != VK_NULL_HANDLE) {
		vkDestroyPipeline(rtg.device, handle, nullptr);
		handle = VK_NULL_HANDLE;
	}

	if (layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(rtg.device, layout, nullptr);
		layout = VK_NULL_HANDLE;
	}

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
	}

	if (set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set_layout, nullptr);
		set_layout = VK_NULL_HANDLE;
	}

	set = VK_NULL_HANDLE;
}


