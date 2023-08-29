#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D ControlSampler;
uniform sampler2D FlashlightSampler;

uniform vec4 ColorModulate;

uniform mat4 ProjMat;
uniform vec2 InSize;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float _FOV;
uniform float Time;

in vec2 texCoord;
out vec4 fragColor;

float near = 0.1; 
float far  = 1000.0;
float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0;
    return (near * far) / (far + near - z * (far - near));    
}

float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 p){
	vec2 ip = floor(p);
	vec2 u = fract(p);
	u = u*u*(3.0-2.0*u);
	
	float res = mix(
		mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
		mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
	return res*res;
}


// Flashlight Variables
const float exposure = 2.;
const float AOE = 8.;

void main() {
    vec4 prev_color = texture(DiffuseSampler, texCoord);


	fragColor = prev_color;

    vec2 uv = texCoord;
    vec2 middle = vec2(0.5, 0.5);
    float distanceToMiddle = abs(distance(uv, middle)) - 0.2;

    // Channel #1
    // Stunt effect 
    vec4 control_color = texelFetch(ControlSampler, ivec2(0, 1), 0);
    int effectTimer = int(control_color.b * 255.);
    if (effectTimer > 0 && effectTimer < 255) {
        float timeSpan = float(effectTimer) / 255.0;


        float brightness = min(0.86, smoothstep(0.0, 1.0, timeSpan) * (noise(texCoord * 6.0) + 1.6));

        fragColor = fragColor + vec4(brightness, brightness, brightness, 1.0);
        // fragColor = mix(
        //     fragColor, 
        //     vec4(1.0, .97, 1.0, 1.0),
        //     (noise(texCoord * 12.0) + 1.6) * 0.3 * smoothstep(0.0, 1.0, timeSpan) * distanceFromMiddle * 4
        // );
    }

    // Channel #2
    // bleeding effect
    control_color = texelFetch(ControlSampler, ivec2(0, 2), 0);
    switch(int(control_color.b * 255.)) {
        case 3:
            distanceToMiddle = distanceToMiddle + 0.3;
        case 2:
            float noiseValue = noise(texCoord * 8.0);
            fragColor = mix(
                fragColor, 
                vec4(1.0, 0.0, 0.0, 1.0),
                distanceToMiddle * noiseValue
            );
            break;
    }

    // Channel #3
    // Soul pickup effect
    control_color = texelFetch(ControlSampler, ivec2(0, 3), 0);
    int effectValue = int(control_color.b * 255.);

    if (effectValue != 0) {
        float intensity = max(0.0, 0.5 - uv.y);
        fragColor = mix(
            fragColor, 
            vec4(103.0 / 255, 255.0 / 255.0, 244.0 / 255.0, 1.0), 
            intensity * sin(max(0.4, control_color.b) * 3.14)
        );
    }

    //Channel #4
    // Cursed forest nausea
    control_color = texelFetch(ControlSampler, ivec2(0, 4), 0);
    int nauseaValue = int(control_color.b * 255.);
    if (nauseaValue == 255) {
        // fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        fragColor = fragColor + vec4(0.01, -0.1, 0.05, 1.0);
    }
    if (nauseaValue > 0 && nauseaValue < 255) {
        float range = 0.01;
        vec4 illusion = min(
            vec4(0.7, 0.7, 0.7, 1.0),
            (texture(DiffuseSampler, texCoord)
            + texture(DiffuseSampler, texCoord + vec2(-range, -range))
            + texture(DiffuseSampler, texCoord + vec2(+range, -range))
            + texture(DiffuseSampler, texCoord + vec2(-range, +range))
            + texture(DiffuseSampler, texCoord + vec2(range, range))) * 0.6
        );
        fragColor = illusion;
    }

}