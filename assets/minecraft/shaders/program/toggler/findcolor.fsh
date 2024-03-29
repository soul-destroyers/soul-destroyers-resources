#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D ParticleSampler;

uniform float Time;

in vec2 texCoord;

out vec4 fragColor;

void readMarker(inout vec4 fragColor, vec4 lastValue, ivec2 markerPos, vec2 markerColor, int row) {
    if (int(gl_FragCoord.x) == 0) {
        vec4 marker = texelFetch(ParticleSampler, markerPos, 0);
        if (marker.rg * 255. == markerColor) {
            fragColor = marker;
        }
    } else {
        vec4 target = texelFetch(DiffuseSampler, ivec2(0, row), 0);
        fragColor = lastValue + sign(target - lastValue)/255.;
    }
}

void main() {
    vec4 lastValue = texture(DiffuseSampler, texCoord);
    fragColor = lastValue;
    switch (int(gl_FragCoord.y)) {
        case 0:
            // Row 0: Time
            float time1 = lastValue.y + (floor(lastValue.x*255.) > ceil(Time*255.) ? 1./255. : 0.0);

            float time2 = lastValue.z + floor(time1)/255.;

            fragColor = vec4(Time, fract(time1), fract(time2), 1);
            break;
        case 1:
            // Row 1: 
            readMarker(fragColor, lastValue, ivec2(0, 0), vec2(254., 253.), 1);
            break;
        case 2:
            // Row 2:
            readMarker(fragColor, lastValue, ivec2(0, 2), vec2(254., 252.), 2);
            break;
        case 3:
            // Row 3:
            readMarker(fragColor, lastValue, ivec2(0, 4), vec2(254., 251.), 3);
            break;
        case 4:
            // Row 4:
            readMarker(fragColor, lastValue, ivec2(0, 6), vec2(254., 250.), 4);
            break;
        case 5:
            // Row 5:
            readMarker(fragColor, lastValue, ivec2(0, 8), vec2(254., 249.), 5);
            break;
        case 6:
            //row 6
            readMarker(fragColor, lastValue, ivec2(0, 10), vec2(254., 248.), 6);
            break;
        case 7:
            //row 7
            readMarker(fragColor, lastValue, ivec2(0, 12), vec2(254., 247.), 7);
            break;
        case 8:
            //row 8
            readMarker(fragColor, lastValue, ivec2(0, 14), vec2(254., 246.), 8);
            break;
    }
}