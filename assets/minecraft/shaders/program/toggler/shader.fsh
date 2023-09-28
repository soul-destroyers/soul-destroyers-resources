#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D ControlSampler;
uniform sampler2D FlashlightSampler;
uniform sampler2D LeavesSampler;

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

float getColorDifference(vec4 color) {
    return color.r + color.g + color.b;
}

// Flashlight Variables
const float exposure = 2.;
const float AOE = 8.;

void main() {
    vec4 prev_color = texture(DiffuseSampler, texCoord);


	fragColor = prev_color;

    //Just a playground
    // float xDistanceToMiddle = abs(texCoord.x - 0.5) + 0.5;
    // float depthValue = max(0.0, (LinearizeDepth(texture2D(DiffuseDepthSampler, texCoord).r) / (sin(Time) * (60 / xDistanceToMiddle))));
    // float dist = (1.0 - depthValue) / 200.0;
    // // vec4 colorUp = texture(DiffuseSampler, texCoord + vec2(0, dist));
    // // vec4 colorDown = texture(DiffuseSampler, texCoord - vec2(0, dist));
    // // vec4 colorLeft = texture(DiffuseSampler, texCoord - vec2(dist, 0));
    // // vec4 colorRight = texture(DiffuseSampler, texCoord + vec2(dist, 0));


    // // float colorDifferenceLevel = 
    // //     getColorDifference(abs(colorUp - colorDown)) 
    // //     + getColorDifference(abs(colorLeft - colorRight));
    // // fragColor = vec4(1.0, 1.0, 1.0, colorDifferenceLevel);
    // // fragColor = vec4(1.0, 1.0, 1.0, depthValue);
    // //end of a playground
    // float currentEffectValue = (smoothstep(0.6, 0.7, 1.0 - depthValue)) * (1.0 - Time) * 0.5;
    // fragColor = vec4(fragColor.r, fragColor.g + currentEffectValue, fragColor.b + currentEffectValue * 2.0, 1.0);

    vec2 uv = texCoord;
    vec2 middle = vec2(0.5, 0.5);
    float distanceToMiddle = abs(distance(uv, middle)) - 0.2;

    // Channel #1
    // Hiding effect
    vec4 control_color = texelFetch(ControlSampler, ivec2(0, 1), 0);
    int effectTimer = int(control_color.b * 255.);
    if (effectTimer > 0 && effectTimer <= 255) {
        float timeSpan = float(effectTimer) / 255.0;


        // float brightness = min(0.86, smoothstep(0.0, 1.0, timeSpan) * (noise(texCoord * 6.0) + 1.6));
        fragColor = mix(
            fragColor,
            vec4(0.0, 0.0, 0.0, 1.0),
            min(0.0, (distanceToMiddle * 0.5) + (0.5 - timeSpan * 1.0))
        );
        vec4 leafColor = texture(LeavesSampler, 1.0 - texCoord);
        float isLeaf = step(0.5, leafColor.a);
        float leafTexture = noise(texCoord * 25.0) * noise(texCoord * 100.0);
        float leafTextureColor = isLeaf * leafTexture;
        fragColor = mix(
            fragColor,
            vec4(0.0, 0.03 + leafTextureColor * 0.05, 0.0, 1.0),
            max(0.0, (leafColor.a * 1.0) + (0.3 - timeSpan * 1.5))
        );
        // stunt effect
        // fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        // fragColor = fragColor + vec4(brightness, brightness, brightness, 1.0);
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
    if (nauseaValue > 0) {
        // fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        fragColor = fragColor + vec4(0.01, -0.1, 0.05, 1.0);
        // vec4 near = texture(DiffuseSampler, texCoord + vec2(-0.02, -0.02));
        // fragColor = mix(
        //     fragColor,
        //     vec4(fragColor.r, near.g + 0.1, near.b + 0.2, 1.0),
        //     0.3
        // );
        // fragColor = fragColor * vec4(vec3(fragColor.g + fragColor.b * 1.3), 1.0);
    }
    if (nauseaValue > 0 && nauseaValue < 255) {
        float range = sin(control_color.b * 3.14) * 0.02;
        vec4 illusion = min(
            vec4(0.7, 0.7, 0.7, 1.0),
            (texture(DiffuseSampler, texCoord)
            + texture(DiffuseSampler, texCoord + vec2(-range, -range))
            + texture(DiffuseSampler, texCoord + vec2(+range, -range))
            + texture(DiffuseSampler, texCoord + vec2(-range, +range))
            + texture(DiffuseSampler, texCoord + vec2(range, range))) * 0.2 + vec4(0.01, -0.1, 0.05, 1.0)
        );
        fragColor = illusion;
    }


    //Channel #5
    // Main menu effect
    control_color = texelFetch(ControlSampler, ivec2(0, 5), 0);
    int desaturationValue = int(control_color.b * 255.);
    // fragColor = vec4(control_color.b, 0.0, 0.0, 1.0);
    if (desaturationValue == 255) {
        fragColor = fragColor * vec4(vec3(fragColor.g + fragColor.b * 1.3), 1.0);
    }

    // float depth = LinearizeDepth(texture2D(DiffuseDepthSampler, texCoord).r);
    // fragColor = vec4(fragColor.rgb / max(1.0, depth), fragColor.a);
    // fragColor = 

    //Channel #6
    // Screen darkening
    control_color = texelFetch(ControlSampler, ivec2(0, 6), 0);
    effectValue = int(control_color.b * 255.);
    if (effectValue > 0 && effectValue < 255) {
        fragColor = vec4(fragColor.rgb, control_color.b);
    }
}