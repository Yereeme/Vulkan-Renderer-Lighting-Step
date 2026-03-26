#include "tutorial.hpp"

#include "Helpers.hpp"
#include "VK.hpp"

static uint32_t vert_code[] =
#include "spv/mirror.vert.inl"
;

static uint32_t frag_code[] =
#include "spv/mirror.frag.inl"
;

void Tutorial::MirrorPipeline::create(
	RTG& rtg,
	VkRenderPass render_pass,
	uint32_t subpass,
	VkDescriptorSetLayout background_set0_env_layout,
	VkDescriptorSetLayout objects_set1_transforms_layout
) {
	VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);
	VkShaderModule frag_module = rtg.helpers.create_shader_module(frag_code);

	// --- push constants (fragment) ---
	VkPushConstantRange pcr{
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(Push),
	};

	// --- pipeline layout: set0(env from background) + set1(transforms from objects) + push constants ---
	{
		std::array<VkDescriptorSetLayout, 2> layouts{
			background_set0_env_layout,     // set = 0 (samplerCube envTex)
			objects_set1_transforms_layout, // set = 1 (Transforms SSBO)
		};

		VkPipelineLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(layouts.size()),
			.pSetLayouts = layouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcr,
		};

		VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
	}

	// ---- graphics pipeline (same general setup as ObjectsPipeline) ----
	std::array<VkPipelineColorBlendAttachmentState, 1> attachment_states{
		VkPipelineColorBlendAttachmentState{
			.blendEnable = VK_FALSE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
				VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		},
	};
	VkPipelineColorBlendStateCreateInfo color_blend_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = uint32_t(attachment_states.size()),
		.pAttachments = attachment_states.data(),
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

	VkGraphicsPipelineCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = uint32_t(stages.size()),
		.pStages = stages.data(),
		.pVertexInputState = &Vertex::array_input_state, // PosNorTexVertex
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

	// modules no longer needed:
	vkDestroyShaderModule(rtg.device, frag_module, nullptr);
	vkDestroyShaderModule(rtg.device, vert_module, nullptr);
}

void Tutorial::MirrorPipeline::destroy(RTG& rtg) {
	//  Mirror does NOT own descriptor set layouts (we reuse background/objects layouts)

	if (layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(rtg.device, layout, nullptr);
		layout = VK_NULL_HANDLE;
	}

	if (handle != VK_NULL_HANDLE) {
		vkDestroyPipeline(rtg.device, handle, nullptr);
		handle = VK_NULL_HANDLE;
	}
}