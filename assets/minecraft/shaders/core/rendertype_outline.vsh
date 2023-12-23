#version 150

in vec3 Position;
in vec4 Color;
in vec2 UV0;
uniform vec4 ColorModulator;


uniform mat4 ModelViewMat;
uniform mat4 ProjMat;

out vec4 vertexColor;
out vec2 texCoord0;
flat out int isMarker;

vec2[] corners = vec2[](
    vec2(0, 1),
    vec2(0, 0),
    vec2(1, 0),
    vec2(1, 1)
);

void main() {

    if (
      abs(Color.r * 255 - 85) < .1
      && abs(Color.g * 255 - 85) < .1
      && abs(Color.b * 255 - 255) < .1
    ) {
      vec2 screenPos = 0.125 * corners[gl_VertexID % 4] - 1.0;
      gl_Position = vec4(screenPos, 0.0, 1.0);
      texCoord0 = vec2(0);
      vertexColor = vec4(0);
      isMarker = 1;
    } else {
      vertexColor = Color;
      texCoord0 = UV0;
      isMarker = 0;
      gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    }

}
