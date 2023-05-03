#version 450

layout (location = 0) out vec3 outColor;

void main() {

  const vec3 vert_pos[3] = vec3[3] (
    vec3(0.5, 0.5, 0.0),
    vec3(-0.5, 0.5, 0.0),
    vec3(0.0, -0.5, 0.0)
  );
  const vec3 vert_col[3] = vec3[3] (
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
  );

  gl_Position = vec4(vert_pos[gl_VertexIndex], 1.0);
  outColor = vert_col[gl_VertexIndex];

}
