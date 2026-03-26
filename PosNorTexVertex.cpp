#include "PosNorTexVertex.hpp"

#include <array>

// one vertex buffer binding (binding=0), tightly packed PosNorTexVertex[]
static std::array< VkVertexInputBindingDescription, 1 > bindings{
	VkVertexInputBindingDescription{
		.binding = 0,
		.stride = sizeof(PosNorTexVertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	}
};

// vertex attributes = what each shader location reads from the struct
static std::array< VkVertexInputAttributeDescription, 4 > attributes{
	VkVertexInputAttributeDescription{
		.location = 0, // Position
		.binding = 0,
		.format = VK_FORMAT_R32G32B32_SFLOAT,
		.offset = offsetof(PosNorTexVertex, Position),
	},
	VkVertexInputAttributeDescription{
		.location = 1, // Normal
		.binding = 0,
		.format = VK_FORMAT_R32G32B32_SFLOAT,
		.offset = offsetof(PosNorTexVertex, Normal),
	},
	VkVertexInputAttributeDescription{
		.location = 2, // MUST BE TEXCOORD (vec2)
		.binding = 0,
		.format = VK_FORMAT_R32G32_SFLOAT,
		.offset = offsetof(PosNorTexVertex, TexCoord),
	},
	VkVertexInputAttributeDescription{
		.location = 3, // MUST BE TANGENT (vec4)
		.binding = 0,
		.format = VK_FORMAT_R32G32B32A32_SFLOAT,
		.offset = offsetof(PosNorTexVertex, Tangent),
	},
};

const VkPipelineVertexInputStateCreateInfo PosNorTexVertex::array_input_state{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	.vertexBindingDescriptionCount = uint32_t(bindings.size()),
	.pVertexBindingDescriptions = bindings.data(),
	.vertexAttributeDescriptionCount = uint32_t(attributes.size()),
	.pVertexAttributeDescriptions = attributes.data(),
};