#version 150

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;

in vec4 vertexColor;
in vec2 texCoord0;
flat in int isMarker;

out vec4 fragColor;

void main() {
  vec4 color = texture(Sampler0, texCoord0);
  ivec2 iCoord = ivec2(gl_FragCoord.xy);
  // if the color of the outline is blue, this means that the outlined entity is put there to get custom shader behaviour
  if (isMarker == 1 && iCoord.x == 0 && iCoord.y == 0) {
    fragColor = vec4(vec3(254, 255, 255) / 255, 1.0);
  } else {
    if (color.a == 0.0) {
        discard;
    }

    fragColor = vec4(ColorModulator.rgb * vertexColor.rgb, ColorModulator.a);
  }
    // fragColor = vec4(vec3(254, 255, 255) / 255, 1.0);
}
