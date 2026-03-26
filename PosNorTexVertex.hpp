#pragma once

#include <vulkan/vulkan_core.h>
#include <cstdint>

struct PosNorTexVertex {
	// basic mesh data:
	struct { float x, y, z; } Position;
	struct { float x, y, z; } Normal;
	struct { float s, t; } TexCoord;

	// tangent frame helper:
	// xyz = tangent direction (T)
	// w   = handedness sign (+1 or -1) so we can rebuild B in shader
	struct { float x, y, z, w; } Tangent;

	// a pipeline vertex input state that works with a buffer holding a PosNorTexVertex[]
	static const VkPipelineVertexInputStateCreateInfo array_input_state;
};

static_assert(sizeof(PosNorTexVertex) == (3 + 3 + 2 + 4) * 4, "PosNorTexVertex is packed.");