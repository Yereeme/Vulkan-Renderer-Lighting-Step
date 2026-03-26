#version 450

layout(location = 0) in vec4 color;

layout(location = 0) out vec4 outColor; //specify color output to the fragment shader, location 0 is first color output of render pass

void main() {
	outColor =  color;
}