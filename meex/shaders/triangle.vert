#version 450

layout (location = 0) out vec3 outColor;

void main() {

  const vec3 vert_pos[3] = vec3[3] (
    vec3(0.5f, 0.5f, 0.0f),
    vec3(-0.5f, 0.5f, 0.0f),
    vec3(0.0f, -0.5f, 0.0f)
  );
  const vec3 vert_col[3] = vec3[3] (
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f)
  );

  gl_Position = vec4(vert_pos[gl_VertexIndex], 1.0f);
  outColor = vert_col[gl_VertexIndex];

}
