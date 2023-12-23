#version 330

uniform sampler2D DiffuseSampler;
uniform sampler2D DepthSampler;
uniform sampler2D OutlineSampler;

uniform vec2 OutSize;
uniform float Time;

in vec2 texCoord;
in vec2 OneTexel;
in vec2 ratio;
in vec3 position;
in mat3 viewmat;
in vec2 proj;
flat in int nobjs;
flat in float GameTime;
flat in int isRenderer;
flat in float controlVal;

out vec4 fragColor;

#define AA 1

#define renderdistance 80
#define fogstart 80

#define PI 3.14159265358979323846



//	Classic Perlin 3D Noise 
//	by Stefan Gustavson
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
vec3 fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

float cnoise(vec3 P){
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod(Pi0, 289.0);
  Pi1 = mod(Pi1, 289.0);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 / 7.0;
  vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 / 7.0;
  vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

#define FPRECISION 2000000.0
int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}
float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}
#define near 0.05
#define far  1000.0
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

float InverseLerp(float currentValue, float minValue, float maxValue) {
    return (currentValue - minValue) / (maxValue - minValue);
}

float Remap(float currentValue, float inMin, float inMax, float outMin, float outMax) {
    float t = InverseLerp(currentValue, inMin, inMax);
    return mix(outMin, outMax, t);
}

//--------------------------------------------------------------------------------
//sdfs
struct obj {float depth; int type;};
obj Plane   (vec3 p,                int type) {return obj(p.y, type);}
obj Sphere  (vec3 p, float r,       int type) {return obj(length(p)-r, type);}
obj Cube    (vec3 p, vec3 b,        int type) {
    vec3 d = abs(p) - b;
    return obj(max(d.x, max(d.y, d.z)), type);
}
//--------------------------------------------------------------------------------
//operations
obj Add(obj a, obj b) {return a.depth < b.depth? a : b;}
obj Sub(obj a, obj b) {return -a.depth > b.depth? a : b;}
obj Intersect(obj a, obj b) {return a.depth > b.depth? a : b;}
obj SmoothAdd(obj a, obj b, float k) {
    float h = clamp(0.5 + 0.5*(b.depth-a.depth)/k, 0.0, 1.0);
    return obj(mix(b.depth, a.depth, h) - k*h*(1.0-h), a.type);
}
obj SmoothSub(obj a, obj b, float k) {
    float h = clamp(0.5 - 0.5*(a.depth+b.depth)/k, 0.0, 1.0);
    return obj(mix(a.depth, -b.depth, h) + k*h*(1.0-h), a.type);
}
obj SmoothIntersect(obj a, obj b, float k) {
    float h = clamp(0.5 - 0.5*(b.depth-a.depth)/k, 0.0, 1.0);
    return obj(mix(b.depth, a.depth, h) + k*h*(1.0-h), a.type);
}
//--------------------------------------------------------------------------------
//scene
obj hit(in vec3 pos) {//obj     pos                     size                    material    smoothness
    obj o = Sphere( pos - vec3(0,64.5,0),    17,                      3);
    // o = SmoothSub(o,    Sphere( pos + vec3(-2,1.5,0),   2,                      1),         0.5);
    // o = Add(o,          Sphere( pos + vec3(2,1,4),      1,                      2));
    // o = Add(o,          Cube(   pos + vec3(5,1,1),      vec3(1),                3));
    // o = SmoothAdd(o,    Sphere( pos + vec3(5.5,0.5,.5), 1,                      3),         0.5);

    //add 20 spheres
    // for (int i = 0; i < 20; i++) {
    //     float r = 1 + 0.5*sin(i*PI/10.0);
    //     o = Add(o, Sphere(pos + 8*vec3(r*cos(i*PI/10.0), -1, r*sin(i*PI/10.0)) + vec3(-5, 0, -5), r, 2));
    // }

    // for (int i = 0; i < nobjs; i++) {
    //     o = Add(o,  Sphere( pos - position + ((vec3(255.0, 1.0, 1.0 / 255.0) * mat3(
    //                     texelFetch(DiffuseSampler, ivec2(3*i + 3,4), 0).rgb,
    //                     texelFetch(DiffuseSampler, ivec2(3*i + 4,4), 0).rgb,
    //                     texelFetch(DiffuseSampler, ivec2(3*i + 5,4), 0).rgb)) - 128), 1, 2));
    // }
    return o;
}
//--------------------------------------------------------------------------------
//filters
float checkerboard(in vec2 p) {
    vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.01;
    vec2 i = 2.0*(abs(fract((p-0.5*w)/2.0)-0.5)-abs(fract((p+0.5*w)/2.0)-0.5))/w;
    return 0.5 - 0.5*i.x*i.y;
}
float shadows(in vec3 ro, in vec3 rd, in float tmin, in float tmax) {
    float soft = 1.0;
    float ph = renderdistance;
    for (float t = tmin; t < tmax;) {
        float h = hit(ro + rd*t).depth;
        if (h < 0.0001) return 0.;

        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        soft = min(soft, 16*d/max(0,t-y));
        ph = h;

        t += h;
    }
    return soft;
}
float AO(in vec3 pos, in vec3 norm) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i=0; i<5; i++) {
        float h = 0.01 + 0.12*float(i)/5.0;
        float d = hit(pos + norm*h).depth;
        occ += (h-d)*sca;
        sca *= 0.95;
        if(occ > 0.35) break;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * norm.y);
}
//--------------------------------------------------------------------------------
//drawing
vec3 getnormal(in vec3 pos) {
    vec2 e = vec2(0.001,0);
    return normalize(vec3(hit(pos+e.xyy).depth-hit(pos-e.xyy).depth,
                          hit(pos+e.yxy).depth-hit(pos-e.yxy).depth,
                          hit(pos+e.yyx).depth-hit(pos-e.yyx).depth));
}
vec4 render(vec3 ro, vec3 rd, float fardepth, vec3 maincolor) {
    //raymarching
    float t = 0.;
    obj o;
    vec3 currPos = ro;
    for(int i = 0; i < 20; i++) {
        o = hit(ro + t*rd);
        currPos = ro + t * rd;
        //if (h.depth < o.depth) o = h;
        //if hit
        if (o.depth < 0.01) break;
        t += o.depth;
        //exceed far plane
        if (t >= fardepth) break;
    }
    //coloring
    vec3 sky = vec3(0.7, 0.9, 1.1);
    vec3 sunlight = vec3(1.5,1.2,1.0);
    vec3 skylight = vec3(0.1,0.2,0.3);
    vec3 indlight = vec3(0.4,0.3,0.2);
    //fake atmosphere by dimming up
    vec4 color = vec4(maincolor, 0.0);
    vec3 sundir = normalize(vec3(-0.5, 0.6, -0.6));
    float sunamount = clamp(dot(sundir,rd), 0.0, 1.0);
    //scene
    if (t < fardepth) {
        vec3 pos = ro + t*rd;
        vec3 norm = getnormal(pos);
        //materials
        switch(o.type) {
            case 1: //plane
                color = vec4(vec3(checkerboard(pos.xz) + 0.3), 1.0);
                break;
            case 2: //sphere
                color = vec4(0.6, 0.3, 0.4, 1.0);
                color.rgb += dot(norm, sundir) * sunlight;
                break;
            case 3: // the positional fog
                color = vec4(vec3(166, 215, 255) / 255, 0.0);
                vec3 posInside = currPos;
                for (int i = 0; i < 30; i++) {
                    obj curr = hit(posInside);
                    if (curr.depth < 0) {
                        vec3 samplerPos = posInside * 0.3;
                        samplerPos.y -= GameTime * 600;
                        float noiseVal = (cnoise(samplerPos) + 1.) * .5;
                        color.a += noiseVal * 0.009 * mix(0.0, 0.5, abs(min(curr.depth, 0.0)));
                    }
                    posInside += rd / 1;
                }
                color.a *= min(Remap(fardepth - distance(ro, currPos), 0, 10, .1, 1), 1);
                break;
            default: color = vec4((norm + 1) / 2, 1.0);
        }
        //lighting
        float skyamount = clamp(0.5 + 0.5*norm.y, 0.0, 1.0);
        float indamount = clamp(dot(norm, normalize(sundir*vec3(-1.0,0.0,-1.0))), 0.0, 1.0);

        float ao = AO(pos, norm);

        vec3 light = 0.4 * sunlight;
        light += skyamount * skylight * ao;
        light += indamount * indlight * ao;

        //fog
        color.rgb = mix(color.rgb, sky, smoothstep(0,1, clamp((t-fogstart)/(renderdistance-fogstart) ,0,1)));
    }
    //world
    else if (t < renderdistance) {
        color = vec4(maincolor, 0.0);
        //fog
        // color = mix(color, sky, smoothstep(0,1, clamp((fardepth-fogstart)/(renderdistance-fogstart) ,0,1)));
    }
    //sun glare
    // color.rgb += 0.25 * vec3(0.8,0.4,0.2) * pow(sunamount, 4.0);

    //return color
    color = clamp(color, 0.0, 1.0);
    return color;
}
//--------------------------------------------------------------------------------
void main() {
    vec3 maincolor = texture(DiffuseSampler, texCoord).rgb;
    vec4 colorVal = texture(DiffuseSampler, texCoord);
    float depth = linearizeDepth(texture(DepthSampler, texCoord).r);

    vec2 uv = (texCoord * 2 - 1);

    //ray start
    vec3 ro = position;
    vec3 rd = viewmat * vec3(uv/proj,-1);
    //warp depth to fov
    float l = length(rd);
    rd /= l;
    depth = depth * l;

    vec4 outlineColor = texture(OutlineSampler, texCoord);
    
    if (isRenderer == 1) {
        //render
        colorVal = render(ro, rd, min(depth, renderdistance), maincolor);
    } else {
        colorVal = outlineColor;
    }
    // colorVal = vec4(controlVal, vec2(0.0),1.0);
    
    // colorVal = vec4(vec3(sin(GameTime * 24000.)), 1.0);
    // colorVal = vec4(maincolor, 1.0);
    // colorVal = vec4(vec3(depth / 100.0), 1.0);
    // colorVal = vec4(render(ro, rd, min(depth, renderdistance), maincolor), 1.0);

    
    fragColor = colorVal;
}
