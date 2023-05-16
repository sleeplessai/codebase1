#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNorm;
layout (location = 2) in vec2 aTex;

uniform mat4 mvp;
uniform mat4 model;

out vec3 norm;
out vec2 tex_coord;
out vec3 frag_pos;

void main() {
  gl_Position = mvp * vec4(aPos, 1.0);
  norm = vec3(normalize(model * vec4(aNorm, 0.0)));
  tex_coord = aTex;
}
