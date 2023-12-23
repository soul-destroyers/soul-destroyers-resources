#version 330

uniform sampler2D DiffuseSampler;
uniform sampler2D OutlineSampler;

in vec2 texCoord;

out vec4 fragColor;

void main() {
  vec4 value = texture(DiffuseSampler, texCoord);
  fragColor = vec4(vec3(0.0), 1.0);

  ivec2 iCoord = ivec2(gl_FragCoord);

  if (iCoord.x == 0 && iCoord.y == 0) {
    vec4 controlValue = texelFetch(OutlineSampler, iCoord, 0);
    // if the current value of the OutlineSampler has the marker value
    if (abs(controlValue.r * 255 - 254) < .1) {
      fragColor.r = 1;
    }
  }
}