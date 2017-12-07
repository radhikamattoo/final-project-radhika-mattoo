
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

//----------------------------------
// VERTICES
//----------------------------------
VertexBufferObject VBO;
int DNA_VERTICES = 95397;
int ROSE_VERTICES = 95397;
int TOTAL = DNA_VERTICES + ROSE_VERTICES;
// int TOTAL = ROSE_VERTICES;
MatrixXf V(3,TOTAL);

//----------------------------------
// NORMALS
//----------------------------------
VertexBufferObject VBO_N;
MatrixXf N(3,TOTAL);
//----------------------------------
// TEXTURE
//----------------------------------
VertexBufferObject VBO_T;
MatrixXf T(2,TOTAL);

//----------------------------------
// MODEL MATRIX
//----------------------------------
MatrixXf model(4,8);

//----------------------------------
// VIEW/CAMERA MATRIX
//----------------------------------
MatrixXf view(4,4);
float focal_length = 4.4;
Vector3f eye(0.0, 0.0, focal_length); //camera position/ eye position  //e
Vector3f look_at(0.0, 0.0, 0.0); //target point, where we want to look //g
Vector3f up_vec(0.0, 1.0, 0.0); //up vector //t

Vector3f lightPos(-1.0, 1.0, 1.0);
//----------------------------------
// PERSPECTIVE PROJECTION MATRIX
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


vector< Vector3f > dna_out_vertices;
vector< Vector2f > dna_out_uvs;
vector< Vector3f > dna_out_normals;

vector< Vector3f > rose_out_vertices;
vector< Vector2f > rose_out_uvs;
vector< Vector3f > rose_out_normals;

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
void readObjFile(string filename, bool rose)
{
  cout << "Reading OBJ file: " << filename << endl;
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
    int uvIndex = uvIndices[i] - 1;
    int normalIndex = normalIndices[i] - 1;

    Vector3f vertex = temp_vertices[vertexIndex];
    Vector2f uv = temp_uvs[uvIndex];
    Vector3f normal = temp_normals[normalIndex];
    if(rose){
      rose_out_vertices.push_back(vertex);
      rose_out_uvs.push_back(uv);
      rose_out_normals.push_back(normal);
    }else{
      dna_out_vertices.push_back(vertex);
      dna_out_uvs.push_back(uv);
      dna_out_normals.push_back(normal);
    }
  }

}
void initialize(GLFWwindow* window)
{
  VertexArrayObject VAO;
  VAO.init();
  VAO.bind();
  // READ IN PARSED DATA
  VBO.init();
  VBO_N.init();
  VBO_T.init();

  // READ IN OBJ FILES
  filenames.push_back("../data/dna_obj/DNA.obj");
  filenames.push_back("../data/rose_obj/Rose.obj");
  readObjFile(filenames[0], false);
  readObjFile(filenames[1], true);

  // cout << "DNA vertices: " << dna_out_vertices.size() << endl;
  // cout << "DNA normals: " << dna_out_normals.size() << endl;
  // cout << "DNA textures: " << dna_out_uvs.size() << endl;
  // for(int i = 0; i < dna_out_vertices.size(); i++)
  // {
  //   Vector3f v_data = dna_out_vertices[i];
  //   Vector3f n_data = dna_out_normals[i];
  //   Vector2f t_data = dna_out_uvs[i];
  //   V.col(i) << v_data[0], v_data[1], v_data[2];
  //   N.col(i) << n_data[0], n_data[1], n_data[2];
  //   T.col(i) << t_data[0], t_data[1];
  // }
  // int start = dna_out_vertices.size();
  int start = 0;
  cout << "Rose vertices: " << rose_out_vertices.size() << endl;
  cout << "Rose normals: " << rose_out_normals.size() << endl;
  cout << "Rose textures: " << rose_out_uvs.size() << endl;
  for(int i = start; i < (start + rose_out_vertices.size()); i++)
  {
    Vector3f v_data = rose_out_vertices[i];
    Vector3f n_data = rose_out_normals[i];
    Vector2f t_data = rose_out_uvs[i];
    v_data[1] -= 40;
    v_data /= 10;
    V.col(i) << v_data[0], v_data[1], v_data[2];
    N.col(i) << n_data[0], n_data[1], n_data[2];
    T.col(i) << t_data[0], t_data[1];
  }

  // cout << "size of vertices: " << (rose_out_vertices.size() +dna_out_vertices.size() )<< endl;
  // cout << "size of normals: " << (rose_out_normals.size() + dna_out_normals.size()) << endl;
  // cout << "size of textures: " << (rose_out_uvs.size() + dna_out_uvs.size()) << endl;
  VBO.update(V);
  VBO_N.update(N);
  VBO_T.update(T);

  // READ IN NORMALS
  //------------------------------------------
  // MODEL MATRIX
  //------------------------------------------
  model <<
  1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;


  float direction = (PI/180) * 90;
  MatrixXf rotation(4,4);
  rotation <<
  1.,    0.,                  0.,                 0.,
  0.,    cos(direction),   sin(direction),  0.,
  0.,    -sin(direction),  cos(direction),  0.,
  0.,    0.,                  0.,                 1.;

  model.block(0,0,4,4) = model.block(0,0,4,4) * rotation;

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

void changeView(int direction)
{
  float factor = 0.3;

  if(direction == 0){
    cout << "Moving eye to the left" << endl;
    eye[0] -= factor;
  }else if(direction == 1){
    cout << "Moving eye to the right" << endl;
    eye[0] += factor;
  }else if(direction == 2){
    cout << "Moving eye up" << endl;
    eye[1] += factor;
  }else if(direction == 3){
    cout << "Moving eye down" << endl;
    eye[1] -= factor;
  }else if(direction == 4){
    cout << "Moving eye in" << endl;
    eye[2] -= factor;
  }else if(direction == 5){
    cout << "Moving eye out" << endl;
    eye[2] += factor;
  }
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

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if(action == GLFW_RELEASE){
    switch(key){
      case GLFW_KEY_LEFT:{
        cout << "Moving LEFT" << endl;
        changeView(0);
        break;
      }
      case GLFW_KEY_RIGHT:{
        cout << "Moving RIGHT" << endl;
        changeView(1);
        break;
      }
      case GLFW_KEY_UP:{
        cout << "Moving UP" << endl;
        changeView(2);
        break;
      }
      case GLFW_KEY_DOWN:{
        cout << "Moving DOWN" << endl;
        changeView(3);
        break;
      }
      case GLFW_KEY_EQUAL:{
        cout << "Moving IN" << endl;
        changeView(4);
        break;
      }
      case GLFW_KEY_MINUS:{
        cout << "Moving OUT" << endl;
        changeView(5);
        break;
      }
    }
  }
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
                    "in vec3 normal;"
                    "in vec2 texCoord;"
                    "out vec2 TexCoord;"
                    "out vec3 Normal;"
                    "out vec3 FragPos;"
                    "uniform mat4 view;"
                    "uniform mat4 projection;"
                    "uniform mat4 model;"
                    "void main()"
                    "{"
                    "    gl_Position = projection * view * model * vec4(position, 1.0);"
                    "    FragPos = vec3(model * vec4(position, 1.0f));"
                    "    Normal = mat3(transpose(inverse(model))) * normal;"
                    "    TexCoord = texCoord;"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "out vec4 outColor;"
                    "in vec2 TexCoord;"
                    "in vec3 Normal;"
                    "in vec3 FragPos;"
                    "uniform vec3 lightPos;"
                    "uniform vec3 viewPos;"
                    "uniform vec3 objectColor;"
                    "uniform sampler2D ourTexture;"
                    "uniform bool is_rose;"
                    "void main()"
                    "{"
                          //ROSE
                  "       if(is_rose){"
                  "         outColor =  texture(ourTexture, TexCoord) ;"
                          //DNA
                  "       }else{"
                  "           outColor =  texture(ourTexture, TexCoord);"
                  "       }"

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
    program.bindVertexAttribArray("normal",VBO_N);
    program.bindVertexAttribArray("texCoord",VBO_T);

    // UNIFORMS
    glUniform3f(program.uniform("lightPos"), lightPos[0] ,lightPos[1], lightPos[2]);
    glUniform3f(program.uniform("viewPos"), eye[0], eye[1], eye[2]);
    glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(program.uniform("projection"), 1, GL_FALSE, projection.data());


    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();


    unsigned int texture1, texture2;
    // -------------------------
    // LOAD DNA TEXTURE
    // -------------------------
    // texture 1
    // ---------
    cout << "Loading DNA texture" << endl;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    unsigned char *data = stbi_load("../data/dna_obj/dna_texture.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
       glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
       glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
       std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);

    // -------------------------
    // LOAD ROSE TEXTURE
    // -------------------------
    // texture 2
    // ---------
    cout << "Loading Rose texture" << endl;
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    // Set our texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Set texture filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    width = 0;
    height = 0;
    nrChannels = 0;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    data = stbi_load("../data/rose_obj/rose_texture.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
       glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
       glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
       std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 1);


    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);


    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
      // Bind your program
      program.bind();
      // ------
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
      glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
      //--------
      // DNA
      //--------
      // bind textures on corresponding texture units
      // glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.block(0,0,4,4).data());
      // glUniform3f(program.uniform("objectColor"), 0.364304, 0.534819, 0.863924);
      // glUniform1i(program.uniform("is_rose"), false);
      // glUniform1i(program.uniform("ourTexture"), 0);
      // glActiveTexture(GL_TEXTURE0);
      // glBindTexture(GL_TEXTURE_2D, texture1);
      // for(int i = 0; i < dna_out_vertices.size(); i+=3){
      //   glDrawArrays(GL_TRIANGLES, i , 3);
      // }

      //--------
      // ROSE
      //--------
      glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.block(0,4,4,4).data());
      glUniform3f(program.uniform("objectColor"), 0.784314, 0.784314, 0.784314);
      glUniform1i(program.uniform("is_rose"), true);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, texture2);
      glUniform1i(program.uniform("ourTexture"), 1);
      // int start = dna_out_vertices.size();
      int start = 0;
      for(int i = start; i <( start + rose_out_vertices.size()); i+=3){
          glDrawArrays(GL_TRIANGLES, i , 3);
      }
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
