
// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>

#include <fstream>

// For A x = b solver
#include <Eigen/Dense>

#include <Eigen/Geometry>

// IO Stream
#include <iostream>

#include <vector>

// Timer
#include <chrono>

#include <cmath>

#include <math.h>

#define PI 3.14159265

// TEXTURE IMAGE READER
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;
using namespace Eigen;

vector<string> filenames;

// Vertices
VertexBufferObject VBO;
MatrixXf V(3,63709);

// Texture
VertexBufferObject VBO_T;
MatrixXf T(2,3);

MatrixXf view(4,4);
float focal_length = 1;
Vector3f eye(-1.0, 0.0, focal_length); //camera position/ eye position  //e
Vector3f look_at(0.0, 0.0, 0.0); //target point, where we want to look //g
Vector3f up_vec(0.0, 1.0, 0.0); //up vector //t

//----------------------------------
// PERSPECTIVE PROJECTION PARAMETERS
//----------------------------------
MatrixXf projection(4,4);
MatrixXf perspective(4,4);
// FOV angle is hardcoded to 60 degrees
float theta = (PI/180) * 60;

// near and far are hardcoded
float n = -0.1;
float f = -100.;
// right and left
float r;
float l;
// top and bottom
float t;
float b;
float aspect;


vector< Vector3f > out_vertices;
vector< Vector2f > out_uvs;
vector< Vector3f > out_normals;

vector<float> split_face_line(string line, int startIdx)
{
  string extracted;
  vector<float> data;
  int z = startIdx;

  for(int i = z; i <= line.length(); i++){
    char val = line[i];
    if(val == '/' || val == ' '  || i == line.length()){ //convert to int and push
      data.push_back(atof(extracted.c_str()));
      extracted = "";
    }else{
      extracted.push_back(val);
    }
  }
  return data;
}
vector<float> split_line(string line, int startIdx)
{
  string extracted;
  vector<float> data;
  int z = startIdx;

  for(int i = z; i <= line.length(); i++){

    char val = line[i];

    if(val == ' ' || i == line.length()){ // Finished building int
      // Convert to int and push to data vector
      data.push_back(atof(extracted.c_str()));
      extracted = "";
    }else{ // Still building int
      extracted.push_back(val);
    }
  }
  return data;
}
void readObjFile(string filename)
{
  // Data holders
  vector< unsigned int > vertexIndices, uvIndices, normalIndices;
  vector< Vector3f > temp_vertices;
  vector< Vector2f > temp_uvs;
  vector< Vector3f > temp_normals;

  // Create file stream
  string line;
  ifstream stream(filename.c_str());
  getline(stream,line);

  // Parse out beginning comments/new lines
  while(line[0] == '#'|| line.length() == 0){
    getline(stream,line);
  }
  while(stream){
    getline(stream,line);
    if(line.length() == 0) continue;
    if(line[0] == 'v'){
      if(line[1] == 't'){ //TEXTURE
        vector<float> values = split_line(line, 3);
        Vector2f uv(values[0], values[1]);
        temp_uvs.push_back(uv);
      }else if(line[1] == 'n'){ //NORMAL
        vector<float> values = split_line(line, 3);
        Vector3f normal(values[0], values[1], values[2]);
        temp_normals.push_back(normal);
      }else{ //VERTEX
        vector<float> values = split_line(line, 2);
        Vector3f vertex(values[0], values[1], values[2]);
        temp_vertices.push_back(vertex);
      }
    }else if(line[0] == 'f'){ //FACE
        unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
        vector<float> data = split_face_line(line, 2);
        // Vertex, Texture UV, Normal
        for(int i = 0; i < data.size(); i+= 3)
        {
          vertexIndices.push_back(data[i]);
          uvIndices.push_back(data[i + 1]);
          normalIndices.push_back(data[i + 2]);
        }
    } //end face if
  } //end while

  // Use index values to format data correctly
  for(int i = 0; i < vertexIndices.size(); i++)
  {
    int vertexIndex = vertexIndices[i] - 1;
    Vector3f vertex = temp_vertices[vertexIndex];
    out_vertices.push_back(vertex);

    int uvIndex = uvIndices[i] - 1;
    Vector2f uv = temp_uvs[uvIndex];
    out_uvs.push_back(uv);

    int normalIndex = normalIndices[i] - 1;
    Vector3f normal = temp_normals[normalIndex];
    out_normals.push_back(normal);
  }
  cout << "Temp vertices size: " << temp_vertices.size() << endl;

}
void initialize(GLFWwindow* window)
{
  VertexArrayObject VAO;
  VAO.init();
  VAO.bind();

  // READ IN OBJ FILES
  cout << "Reading OBJ files" << endl;
  filenames.push_back("../data/dna_obj/dna.obj");
  readObjFile(filenames[0]);
  cout << "Vertex size: " << out_vertices.size() << endl;
  cout << "Individual vertex size: " << out_vertices[0] << endl;
  // filenames.push_back("../data/rose_obj/rose.obj");
  // readObjFile(filenames[1]);

  // READ IN VERTICES
  VBO.init();
  for(int i = 0; i < out_vertices.size(); i++)
  {
    Vector3f data = out_vertices[i];
    V.col(i) << data[0], data[1], data[2];
  }
  VBO.update(V);


  // VBO_T.init();
  // int start = 0;
  // V.col(start) << 0.5f, 0.5f, 0.0f;
  // V.col(start + 1) <<   0.5, -0.5, 0.5;
  // V.col(start + 2) <<   -0.5,-0.5, 0.0;
  // VBO.update(V);
  // for(int i = 0; i < out_vertices.size(); i++){
  //   // V.col(i)
  // }

  // // READ IN TEXTURE INDICES
  // T.col(start) << 1.0, 1.0;
  // T.col(start + 1) <<  1.0, 0.0;
  // T.col(start + 2) <<   0.0, 0.0;
  // VBO_T.update(T);

  // READ IN NORMALS
  //------------------------------------------
  // VIEW/CAMERA MATRIX
  //------------------------------------------
  Vector3f w = (eye - look_at).normalized();
  Vector3f u = (up_vec.cross(w).normalized());
  Vector3f v = w.cross(u);

  Matrix4f look;
  look <<
  u[0], u[1], u[2], 0.,
  v[0], v[1], v[2], 0.,
  w[0], w[1], w[2], 0.,
  0.,   0.,    0.,  0.5;

  Matrix4f at;
  at <<
  0.5, 0.0, 0.0, -eye[0],
  0.0, 0.5, 0.0, -eye[1],
  0.0, 0.0, 0.5, -eye[2],
  0.0, 0.0, 0.0, 0.5;

  view = look * at;
  //------------------------------------------
    // PROJECTION MATRIX
    //------------------------------------------
    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    aspect = width/height;

    t = tan(theta/2) * abs(n);
    b = -t;

    r = aspect * t;
    l = -r;

    perspective <<
    2*abs(n)/(r-l), 0., (r+l)/(r-l), 0.,
    0., (2 * abs(n))/(t-b), (t+b)/(t-b), 0.,
    0., 0.,  (abs(f) + abs(n))/(abs(n) - abs(f)), (2 * abs(f) * abs(n))/(abs(n) - abs(f)),
    0., 0., -1., 0;

    projection = perspective;


}
int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
  #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  #endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec3 position;" //vertex position
                    // "in vec2 aTexCoord;"
                    // "out vec2 TexCoord;"
                    "uniform mat4 view;"
                    "uniform mat4 projection;"
                    "void main()"
                    "{"
                    "    gl_Position = projection * view * vec4(position, 1.0);"
                    // "     TexCoord = aTexCoord;"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    // "in vec2 TexCoord;"
                    "out vec4 outColor;"
                    // "uniform sampler2D ourTexture;"
                    "void main()"
                    "{"
                        // "outColor = texture(ourTexture, TexCoord);"
                    "    outColor = vec4(0.0, 0.0, 0.0, 0.0);"
                    "}";

    // INITIALIZE EVERYTHING
    initialize(window);

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    // The vertex shader wants the position of the vertices as an input.
    // The following line connects the VBO we defined above with the position "slot"
    // in the vertex shader
    program.bindVertexAttribArray("position",VBO);
    // program.bindVertexAttribArray("aTexCoord",VBO_T);
    glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(program.uniform("projection"), 1, GL_FALSE, projection.data());
    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();


    // // load and create a texture
    //  // -------------------------
    //  unsigned int texture1, texture2;
    //  // texture 1
    //  // ---------
    //  glGenTextures(1, &texture1);
    //  glBindTexture(GL_TEXTURE_2D, texture1);
    //   // set the texture wrapping parameters
    //  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    //  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //  // set texture filtering parameters
    //  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //
    //  // load image, create texture and generate mipmaps
    //  int width, height, nrChannels;
    //  stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    //  // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    //  unsigned char *data = stbi_load("../data/noisewood3.jpg", &width, &height, &nrChannels, 0);
    //  if (data)
    //  {
    //      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    //      glGenerateMipmap(GL_TEXTURE_2D);
    //  }
    //  else
    //  {
    //      std::cout << "Failed to load texture" << std::endl;
    //  }
    //  stbi_image_free(data);
    //  cout << "Setting texture uniform" << endl;
    //  glUniform1i(program.uniform("ourTexture"), 0);


    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
      // Bind your program
      program.bind();
      // ------
      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      // bind textures on corresponding texture units
      // glActiveTexture(GL_TEXTURE0);
      // glBindTexture(GL_TEXTURE_2D, texture1);
      glDrawArrays(GL_LINES, 0, out_vertices.size() );      // Clear the framebuffer

      // Swap front and back buffers
      glfwSwapBuffers(window);

      // Poll for and process events
      glfwPollEvents();

    }
    // Deallocate opengl memory
    program.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
