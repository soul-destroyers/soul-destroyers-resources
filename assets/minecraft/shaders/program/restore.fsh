#version 330

uniform sampler2D DiffuseSampler;
uniform vec2 OutSize;

in vec2 texCoord;
in vec2 oneTexel;

out vec4 fragColor;

#define PIXELATION_FACTOR 2

void main() {
  vec4 colorVal = texture(DiffuseSampler, texCoord);
  ivec2 texelCoord = ivec2(texCoord * OutSize);

  ivec2 dataPixel = ivec2(texelCoord.x % PIXELATION_FACTOR, texelCoord.y % PIXELATION_FACTOR);


  colorVal = texelFetch(DiffuseSampler, texelCoord - dataPixel, 0);

  fragColor = colorVal;
}