#version 150
in vec4 Position;

uniform vec2 InSize;
uniform vec2 OutSize;
uniform vec2 ScreenSize;

uniform sampler2D ControlSampler;

out vec2 texCoord;
out vec2 screenCoord;
// out float daytime;

void main(){
    float poi = texture(ControlSampler,vec2(0.0,0.0)).a;
    // same as blit_copy but you can change if u want
    float x = -1.0; 
    float y = -1.0;

    if (Position.x > 0.001){
        x = 1.0;
    }
    if (Position.y > 0.001){
        y = 1.0;
    }

    gl_Position = vec4(x, y, 0.2, 1.0);
    texCoord = Position.xy / OutSize;
    screenCoord = Position.xy;
    // Channel #8 daytime
    // daytime = texelFetch(ControlSampler, ivec2(0, 8), 0);
}