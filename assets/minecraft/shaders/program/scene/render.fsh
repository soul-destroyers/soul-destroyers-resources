#version 330

uniform sampler2D DiffuseSampler;
uniform sampler2D DepthSampler;
uniform sampler2D OutlineSampler;

uniform vec2 OutSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;
in vec2 ratio;
in vec3 position;
in mat3 viewmat;
in vec2 proj;
flat in int nobjs;
flat in float GameTime;
flat in int isRenderer;
flat in float controlVal;

out vec4 fragColor;

#define PIXELATION_FACTOR 2

#define AA 1

#define renderdistance 80
#define fogstart 80

#define PI 3.14159265358979323846

vec4 FAST32_hash_3D_Cell( vec3 gridcell )	//	generates 4 different random numbers for the single given cell point
{
    //    gridcell is assumed to be an integer coordinate

    //	TODO: 	these constants need tweaked to find the best possible noise.
    //			probably requires some kind of brute force computational searching or something....
    const vec2 OFFSET = vec2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const vec4 SOMELARGEFLOATS = vec4( 635.298681, 682.357502, 668.926525, 588.255119 );
    const vec4 ZINC = vec4( 48.500388, 65.294118, 63.934599, 63.279683 );

    //	truncate the domain
    gridcell.xyz = gridcell - floor(gridcell * ( 1.0 / DOMAIN )) * DOMAIN;
    gridcell.xy += OFFSET.xy;
    gridcell.xy *= gridcell.xy;
    return fract( ( gridcell.x * gridcell.y ) * ( 1.0 / ( SOMELARGEFLOATS + gridcell.zzzz * ZINC ) ) );
}
void FAST32_hash_3D( vec3 gridcell, out vec4 lowz_hash, out vec4 highz_hash )	//	generates a random number for each of the 8 cell corners
{
    //    gridcell is assumed to be an integer coordinate

    //	TODO: 	these constants need tweaked to find the best possible noise.
    //			probably requires some kind of brute force computational searching or something....
    const vec2 OFFSET = vec2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const float SOMELARGEFLOAT = 635.298681;
    const float ZINC = 48.500388;

    //	truncate the domain
    gridcell.xyz = gridcell.xyz - floor(gridcell.xyz * ( 1.0 / DOMAIN )) * DOMAIN;
    vec3 gridcell_inc1 = step( gridcell, vec3( DOMAIN - 1.5 ) ) * ( gridcell + 1.0 );

    //	calculate the noise
    vec4 P = vec4( gridcell.xy, gridcell_inc1.xy ) + OFFSET.xyxy;
    P *= P;
    P = P.xzxz * P.yyww;
    highz_hash.xy = vec2( 1.0 / ( SOMELARGEFLOAT + vec2( gridcell.z, gridcell_inc1.z ) * ZINC ) );
    lowz_hash = fract( P * highz_hash.xxxx );
    highz_hash = fract( P * highz_hash.yyyy );
}
void FAST32_hash_3D( 	vec3 gridcell,
                        vec3 v1_mask,		//	user definable v1 and v2.  ( 0's and 1's )
                        vec3 v2_mask,
                        out vec4 hash_0,
                        out vec4 hash_1,
                        out vec4 hash_2	)		//	generates 3 random numbers for each of the 4 3D cell corners.  cell corners:  v0=0,0,0  v3=1,1,1  the other two are user definable
{
    //    gridcell is assumed to be an integer coordinate

    //	TODO: 	these constants need tweaked to find the best possible noise.
    //			probably requires some kind of brute force computational searching or something....
    const vec2 OFFSET = vec2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const vec3 SOMELARGEFLOATS = vec3( 635.298681, 682.357502, 668.926525 );
    const vec3 ZINC = vec3( 48.500388, 65.294118, 63.934599 );

    //	truncate the domain
    gridcell.xyz = gridcell.xyz - floor(gridcell.xyz * ( 1.0 / DOMAIN )) * DOMAIN;
    vec3 gridcell_inc1 = step( gridcell, vec3( DOMAIN - 1.5 ) ) * ( gridcell + 1.0 );

    //	compute x*x*y*y for the 4 corners
    vec4 P = vec4( gridcell.xy, gridcell_inc1.xy ) + OFFSET.xyxy;
    P *= P;
    vec4 V1xy_V2xy = mix( P.xyxy, P.zwzw, vec4( v1_mask.xy, v2_mask.xy ) );		//	apply mask for v1 and v2
    P = vec4( P.x, V1xy_V2xy.xz, P.z ) * vec4( P.y, V1xy_V2xy.yw, P.w );

    //	get the lowz and highz mods
    vec3 lowz_mods = vec3( 1.0 / ( SOMELARGEFLOATS.xyz + gridcell.zzz * ZINC.xyz ) );
    vec3 highz_mods = vec3( 1.0 / ( SOMELARGEFLOATS.xyz + gridcell_inc1.zzz * ZINC.xyz ) );

    //	apply mask for v1 and v2 mod values
    v1_mask = ( v1_mask.z < 0.5 ) ? lowz_mods : highz_mods;
    v2_mask = ( v2_mask.z < 0.5 ) ? lowz_mods : highz_mods;

    //	compute the final hash
    hash_0 = fract( P * vec4( lowz_mods.x, v1_mask.x, v2_mask.x, highz_mods.x ) );
    hash_1 = fract( P * vec4( lowz_mods.y, v1_mask.y, v2_mask.y, highz_mods.y ) );
    hash_2 = fract( P * vec4( lowz_mods.z, v1_mask.z, v2_mask.z, highz_mods.z ) );
}
vec4 FAST32_hash_3D( 	vec3 gridcell,
                        vec3 v1_mask,		//	user definable v1 and v2.  ( 0's and 1's )
                        vec3 v2_mask )		//	generates 1 random number for each of the 4 3D cell corners.  cell corners:  v0=0,0,0  v3=1,1,1  the other two are user definable
{
    //    gridcell is assumed to be an integer coordinate

    //	TODO: 	these constants need tweaked to find the best possible noise.
    //			probably requires some kind of brute force computational searching or something....
    const vec2 OFFSET = vec2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const float SOMELARGEFLOAT = 635.298681;
    const float ZINC = 48.500388;

    //	truncate the domain
    gridcell.xyz = gridcell.xyz - floor(gridcell.xyz * ( 1.0 / DOMAIN )) * DOMAIN;
    vec3 gridcell_inc1 = step( gridcell, vec3( DOMAIN - 1.5 ) ) * ( gridcell + 1.0 );

    //	compute x*x*y*y for the 4 corners
    vec4 P = vec4( gridcell.xy, gridcell_inc1.xy ) + OFFSET.xyxy;
    P *= P;
    vec4 V1xy_V2xy = mix( P.xyxy, P.zwzw, vec4( v1_mask.xy, v2_mask.xy ) );		//	apply mask for v1 and v2
    P = vec4( P.x, V1xy_V2xy.xz, P.z ) * vec4( P.y, V1xy_V2xy.yw, P.w );

    //	get the z mod vals
    vec2 V1z_V2z = vec2( v1_mask.z < 0.5 ? gridcell.z : gridcell_inc1.z, v2_mask.z < 0.5 ? gridcell.z : gridcell_inc1.z );
    vec4 mod_vals = vec4( 1.0 / ( SOMELARGEFLOAT + vec4( gridcell.z, V1z_V2z, gridcell_inc1.z ) * ZINC ) );

    //	compute the final hash
    return fract( P * mod_vals );
}
void FAST32_hash_3D( 	vec3 gridcell,
                        out vec4 lowz_hash_0,
                        out vec4 lowz_hash_1,
                        out vec4 lowz_hash_2,
                        out vec4 highz_hash_0,
                        out vec4 highz_hash_1,
                        out vec4 highz_hash_2	)		//	generates 3 random numbers for each of the 8 cell corners
{
    //    gridcell is assumed to be an integer coordinate

    //	TODO: 	these constants need tweaked to find the best possible noise.
    //			probably requires some kind of brute force computational searching or something....
    const vec2 OFFSET = vec2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const vec3 SOMELARGEFLOATS = vec3( 635.298681, 682.357502, 668.926525 );
    const vec3 ZINC = vec3( 48.500388, 65.294118, 63.934599 );

    //	truncate the domain
    gridcell.xyz = gridcell.xyz - floor(gridcell.xyz * ( 1.0 / DOMAIN )) * DOMAIN;
    vec3 gridcell_inc1 = step( gridcell, vec3( DOMAIN - 1.5 ) ) * ( gridcell + 1.0 );

    //	calculate the noise
    vec4 P = vec4( gridcell.xy, gridcell_inc1.xy ) + OFFSET.xyxy;
    P *= P;
    P = P.xzxz * P.yyww;
    vec3 lowz_mod = vec3( 1.0 / ( SOMELARGEFLOATS.xyz + gridcell.zzz * ZINC.xyz ) );
    vec3 highz_mod = vec3( 1.0 / ( SOMELARGEFLOATS.xyz + gridcell_inc1.zzz * ZINC.xyz ) );
    lowz_hash_0 = fract( P * lowz_mod.xxxx );
    highz_hash_0 = fract( P * highz_mod.xxxx );
    lowz_hash_1 = fract( P * lowz_mod.yyyy );
    highz_hash_1 = fract( P * highz_mod.yyyy );
    lowz_hash_2 = fract( P * lowz_mod.zzzz );
    highz_hash_2 = fract( P * highz_mod.zzzz );
}
void FAST32_hash_3D( 	vec3 gridcell,
                        out vec4 lowz_hash_0,
                        out vec4 lowz_hash_1,
                        out vec4 lowz_hash_2,
                        out vec4 lowz_hash_3,
                        out vec4 highz_hash_0,
                        out vec4 highz_hash_1,
                        out vec4 highz_hash_2,
                        out vec4 highz_hash_3	)		//	generates 4 random numbers for each of the 8 cell corners
{
    //    gridcell is assumed to be an integer coordinate

    //	TODO: 	these constants need tweaked to find the best possible noise.
    //			probably requires some kind of brute force computational searching or something....
    const vec2 OFFSET = vec2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const vec4 SOMELARGEFLOATS = vec4( 635.298681, 682.357502, 668.926525, 588.255119 );
    const vec4 ZINC = vec4( 48.500388, 65.294118, 63.934599, 63.279683 );

    //	truncate the domain
    gridcell.xyz = gridcell.xyz - floor(gridcell.xyz * ( 1.0 / DOMAIN )) * DOMAIN;
    vec3 gridcell_inc1 = step( gridcell, vec3( DOMAIN - 1.5 ) ) * ( gridcell + 1.0 );

    //	calculate the noise
    vec4 P = vec4( gridcell.xy, gridcell_inc1.xy ) + OFFSET.xyxy;
    P *= P;
    P = P.xzxz * P.yyww;
    lowz_hash_3.xyzw = vec4( 1.0 / ( SOMELARGEFLOATS.xyzw + gridcell.zzzz * ZINC.xyzw ) );
    highz_hash_3.xyzw = vec4( 1.0 / ( SOMELARGEFLOATS.xyzw + gridcell_inc1.zzzz * ZINC.xyzw ) );
    lowz_hash_0 = fract( P * lowz_hash_3.xxxx );
    highz_hash_0 = fract( P * highz_hash_3.xxxx );
    lowz_hash_1 = fract( P * lowz_hash_3.yyyy );
    highz_hash_1 = fract( P * highz_hash_3.yyyy );
    lowz_hash_2 = fract( P * lowz_hash_3.zzzz );
    highz_hash_2 = fract( P * highz_hash_3.zzzz );
    lowz_hash_3 = fract( P * lowz_hash_3.wwww );
    highz_hash_3 = fract( P * highz_hash_3.wwww );
}

float Cellular3D(vec3 P)
{
    //	establish our grid cell and unit position
    vec3 Pi = floor(P);
    vec3 Pf = P - Pi;

    //	calculate the hash.
    //	( various hashing methods listed in order of speed )
    vec4 hash_x0, hash_y0, hash_z0, hash_x1, hash_y1, hash_z1;
    FAST32_hash_3D( Pi, hash_x0, hash_y0, hash_z0, hash_x1, hash_y1, hash_z1 );
    //SGPP_hash_3D( Pi, hash_x0, hash_y0, hash_z0, hash_x1, hash_y1, hash_z1 );

    //	non-weighted jitter window.  jitter window of 0.4 will give results similar to Stefans original implementation
    //	nicer looking, faster, but has minor artifacts.  ( discontinuities in signal )
    const float JITTER_WINDOW = 0.4;
    hash_x0 = hash_x0 * JITTER_WINDOW * 2.0 + vec4(-JITTER_WINDOW, 1.0-JITTER_WINDOW, -JITTER_WINDOW, 1.0-JITTER_WINDOW);
    hash_y0 = hash_y0 * JITTER_WINDOW * 2.0 + vec4(-JITTER_WINDOW, -JITTER_WINDOW, 1.0-JITTER_WINDOW, 1.0-JITTER_WINDOW);
    hash_x1 = hash_x1 * JITTER_WINDOW * 2.0 + vec4(-JITTER_WINDOW, 1.0-JITTER_WINDOW, -JITTER_WINDOW, 1.0-JITTER_WINDOW);
    hash_y1 = hash_y1 * JITTER_WINDOW * 2.0 + vec4(-JITTER_WINDOW, -JITTER_WINDOW, 1.0-JITTER_WINDOW, 1.0-JITTER_WINDOW);
    hash_z0 = hash_z0 * JITTER_WINDOW * 2.0 + vec4(-JITTER_WINDOW, -JITTER_WINDOW, -JITTER_WINDOW, -JITTER_WINDOW);
    hash_z1 = hash_z1 * JITTER_WINDOW * 2.0 + vec4(1.0-JITTER_WINDOW, 1.0-JITTER_WINDOW, 1.0-JITTER_WINDOW, 1.0-JITTER_WINDOW);

    //	return the closest squared distance
    vec4 dx1 = Pf.xxxx - hash_x0;
    vec4 dy1 = Pf.yyyy - hash_y0;
    vec4 dz1 = Pf.zzzz - hash_z0;
    vec4 dx2 = Pf.xxxx - hash_x1;
    vec4 dy2 = Pf.yyyy - hash_y1;
    vec4 dz2 = Pf.zzzz - hash_z1;
    vec4 d1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    vec4 d2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    d1 = min(d1, d2);
    d1.xy = min(d1.xy, d1.wz);
    return min(d1.x, d1.y) * ( 9.0 / 12.0 );	//	scale return value from 0.0->1.333333 to 0.0->1.0  	(2/3)^2 * 3  == (12/9) == 1.333333
}



vec3 permute(vec3 x) {
  return mod((34.0 * x + 1.0) * x, 289.0);
}

vec3 dist(vec3 x, vec3 y, vec3 z,  bool manhattanDistance) {
  return manhattanDistance ?  abs(x) + abs(y) + abs(z) :  (x * x + y * y + z * z);
}

vec2 worley(vec3 P, float jitter, bool manhattanDistance) {
float K = 0.142857142857; // 1/7
float Ko = 0.428571428571; // 1/2-K/2
float  K2 = 0.020408163265306; // 1/(7*7)
float Kz = 0.166666666667; // 1/6
float Kzo = 0.416666666667; // 1/2-1/6*2

	vec3 Pi = mod(floor(P), 289.0);
 	vec3 Pf = fract(P) - 0.5;

	vec3 Pfx = Pf.x + vec3(1.0, 0.0, -1.0);
	vec3 Pfy = Pf.y + vec3(1.0, 0.0, -1.0);
	vec3 Pfz = Pf.z + vec3(1.0, 0.0, -1.0);

	vec3 p = permute(Pi.x + vec3(-1.0, 0.0, 1.0));
	vec3 p1 = permute(p + Pi.y - 1.0);
	vec3 p2 = permute(p + Pi.y);
	vec3 p3 = permute(p + Pi.y + 1.0);

	vec3 p11 = permute(p1 + Pi.z - 1.0);
	vec3 p12 = permute(p1 + Pi.z);
	vec3 p13 = permute(p1 + Pi.z + 1.0);

	vec3 p21 = permute(p2 + Pi.z - 1.0);
	vec3 p22 = permute(p2 + Pi.z);
	vec3 p23 = permute(p2 + Pi.z + 1.0);

	vec3 p31 = permute(p3 + Pi.z - 1.0);
	vec3 p32 = permute(p3 + Pi.z);
	vec3 p33 = permute(p3 + Pi.z + 1.0);

	vec3 ox11 = fract(p11*K) - Ko;
	vec3 oy11 = mod(floor(p11*K), 7.0)*K - Ko;
	vec3 oz11 = floor(p11*K2)*Kz - Kzo; // p11 < 289 guaranteed

	vec3 ox12 = fract(p12*K) - Ko;
	vec3 oy12 = mod(floor(p12*K), 7.0)*K - Ko;
	vec3 oz12 = floor(p12*K2)*Kz - Kzo;

	vec3 ox13 = fract(p13*K) - Ko;
	vec3 oy13 = mod(floor(p13*K), 7.0)*K - Ko;
	vec3 oz13 = floor(p13*K2)*Kz - Kzo;

	vec3 ox21 = fract(p21*K) - Ko;
	vec3 oy21 = mod(floor(p21*K), 7.0)*K - Ko;
	vec3 oz21 = floor(p21*K2)*Kz - Kzo;

	vec3 ox22 = fract(p22*K) - Ko;
	vec3 oy22 = mod(floor(p22*K), 7.0)*K - Ko;
	vec3 oz22 = floor(p22*K2)*Kz - Kzo;

	vec3 ox23 = fract(p23*K) - Ko;
	vec3 oy23 = mod(floor(p23*K), 7.0)*K - Ko;
	vec3 oz23 = floor(p23*K2)*Kz - Kzo;

	vec3 ox31 = fract(p31*K) - Ko;
	vec3 oy31 = mod(floor(p31*K), 7.0)*K - Ko;
	vec3 oz31 = floor(p31*K2)*Kz - Kzo;

	vec3 ox32 = fract(p32*K) - Ko;
	vec3 oy32 = mod(floor(p32*K), 7.0)*K - Ko;
	vec3 oz32 = floor(p32*K2)*Kz - Kzo;

	vec3 ox33 = fract(p33*K) - Ko;
	vec3 oy33 = mod(floor(p33*K), 7.0)*K - Ko;
	vec3 oz33 = floor(p33*K2)*Kz - Kzo;

	vec3 dx11 = Pfx + jitter*ox11;
	vec3 dy11 = Pfy.x + jitter*oy11;
	vec3 dz11 = Pfz.x + jitter*oz11;

	vec3 dx12 = Pfx + jitter*ox12;
	vec3 dy12 = Pfy.x + jitter*oy12;
	vec3 dz12 = Pfz.y + jitter*oz12;

	vec3 dx13 = Pfx + jitter*ox13;
	vec3 dy13 = Pfy.x + jitter*oy13;
	vec3 dz13 = Pfz.z + jitter*oz13;

	vec3 dx21 = Pfx + jitter*ox21;
	vec3 dy21 = Pfy.y + jitter*oy21;
	vec3 dz21 = Pfz.x + jitter*oz21;

	vec3 dx22 = Pfx + jitter*ox22;
	vec3 dy22 = Pfy.y + jitter*oy22;
	vec3 dz22 = Pfz.y + jitter*oz22;

	vec3 dx23 = Pfx + jitter*ox23;
	vec3 dy23 = Pfy.y + jitter*oy23;
	vec3 dz23 = Pfz.z + jitter*oz23;

	vec3 dx31 = Pfx + jitter*ox31;
	vec3 dy31 = Pfy.z + jitter*oy31;
	vec3 dz31 = Pfz.x + jitter*oz31;

	vec3 dx32 = Pfx + jitter*ox32;
	vec3 dy32 = Pfy.z + jitter*oy32;
	vec3 dz32 = Pfz.y + jitter*oz32;

	vec3 dx33 = Pfx + jitter*ox33;
	vec3 dy33 = Pfy.z + jitter*oy33;
	vec3 dz33 = Pfz.z + jitter*oz33;

	vec3 d11 = dist(dx11, dy11, dz11, manhattanDistance);
	vec3 d12 =dist(dx12, dy12, dz12, manhattanDistance);
	vec3 d13 = dist(dx13, dy13, dz13, manhattanDistance);
	vec3 d21 = dist(dx21, dy21, dz21, manhattanDistance);
	vec3 d22 = dist(dx22, dy22, dz22, manhattanDistance);
	vec3 d23 = dist(dx23, dy23, dz23, manhattanDistance);
	vec3 d31 = dist(dx31, dy31, dz31, manhattanDistance);
	vec3 d32 = dist(dx32, dy32, dz32, manhattanDistance);
	vec3 d33 = dist(dx33, dy33, dz33, manhattanDistance);

	vec3 d1a = min(d11, d12);
	d12 = max(d11, d12);
	d11 = min(d1a, d13); // Smallest now not in d12 or d13
	d13 = max(d1a, d13);
	d12 = min(d12, d13); // 2nd smallest now not in d13
	vec3 d2a = min(d21, d22);
	d22 = max(d21, d22);
	d21 = min(d2a, d23); // Smallest now not in d22 or d23
	d23 = max(d2a, d23);
	d22 = min(d22, d23); // 2nd smallest now not in d23
	vec3 d3a = min(d31, d32);
	d32 = max(d31, d32);
	d31 = min(d3a, d33); // Smallest now not in d32 or d33
	d33 = max(d3a, d33);
	d32 = min(d32, d33); // 2nd smallest now not in d33
	vec3 da = min(d11, d21);
	d21 = max(d11, d21);
	d11 = min(da, d31); // Smallest now in d11
	d31 = max(da, d31); // 2nd smallest now not in d31
	d11.xy = (d11.x < d11.y) ? d11.xy : d11.yx;
	d11.xz = (d11.x < d11.z) ? d11.xz : d11.zx; // d11.x now smallest
	d12 = min(d12, d21); // 2nd smallest now not in d21
	d12 = min(d12, d22); // nor in d22
	d12 = min(d12, d31); // nor in d31
	d12 = min(d12, d32); // nor in d32
	d11.yz = min(d11.yz,d12.xy); // nor in d12.yz
	d11.y = min(d11.y,d12.z); // Only two more to go
	d11.y = min(d11.y,d11.z); // Done! (Phew!)
	return sqrt(d11.xy); // F1, F2

}


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
    obj o = Sphere( pos - vec3(0,64,0),    150,                      3);

    o = Add(o, Sphere(pos - vec3(100,20,0), 150, 3));
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
        if (o.depth < 0.01) break;
        //if hit
        t += o.depth;
        //exceed far plane
        if (t >= fardepth) break;
    }

    vec3 sky = vec3(0.7, 0.9, 1.1);
    vec3 sunlight = vec3(1.5,1.2,1.0);

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
            case 3: // white spider celestial web
                color = vec4(vec3(140, 140, 180) / 255, 0.0);
                vec3 posInside = currPos;
                for (int i = 0; i < 12; i++) {
                    obj curr = hit(posInside);
                    if (curr.depth < 0) {
                        vec3 samplerPos = posInside * .27;
                        samplerPos.x += GameTime * 300;
                        float noise = clamp(Cellular3D(samplerPos), 0.15, .34);
                        float colorVal = (noise) * 0.253 + .015;
                        color.a += colorVal;
                    }
                    posInside += rd;
                }
                color.a *= min(Remap(fardepth - distance(ro, currPos), 0, 10, .1, 1), 1);
                break;
            default: color = vec4((norm + 1) / 2, 1.0);
        }
        //fog
        color.rgb = mix(color.rgb, sky, smoothstep(0,1, clamp((t-fogstart)/(renderdistance-fogstart) ,0,1)));
    }
    //world
    else if (t < renderdistance) {
        color = vec4(maincolor, 0.0);
    }

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
    
    bool isDataPixel = int((texCoord * OutSize).x) % PIXELATION_FACTOR == 0;

    if (isRenderer == 1) {
        //render
        if (isDataPixel) {
            colorVal = render(ro, rd, min(depth, renderdistance), maincolor);
        } else {
            colorVal = vec4(0.0);
        }
    } else {
        colorVal = outlineColor;
    }
    
    fragColor = colorVal;
}
