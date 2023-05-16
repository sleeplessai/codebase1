#version 330 core

out vec4 outColor;

struct material_tex2d_t {
  sampler2D texture_ambient1;
  sampler2D texture_diffuse1;
  sampler2D texture_specular1;
};

struct directional_light_t {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  vec3 direction;
};

uniform material_tex2d_t mtl;
uniform directional_light_t lt;
uniform vec3 view_pos;
uniform bool _Normal_as_color;
in vec3 norm;
in vec2 tex_coord;
in vec3 frag_pos;

void main() {
  vec3 ambient = lt.ambient * texture(mtl.texture_ambient1, tex_coord).rgb;

  vec3 lt_dir = normalize(-lt.direction);
  vec3 norm_dir = normalize(norm);
  float diff = max(dot(lt_dir, norm_dir), 0.0);
  vec3 diffuse = lt.diffuse * diff * texture(mtl.texture_diffuse1, tex_coord).rgb;

  vec3 ref_dir = normalize(reflect(-lt_dir, norm_dir));
  vec3 view_dir = normalize(view_pos - frag_pos);
  float spec = pow(max(dot(view_dir, ref_dir), 0.0), 32.0);
  vec3 specular = lt.specular * spec * texture(mtl.texture_specular1, tex_coord).rgb;

  outColor = vec4(ambient + diffuse + specular, 1.0);

  if (_Normal_as_color) {
    outColor = vec4(norm + vec3(0.1), 1.0);
  }
}
