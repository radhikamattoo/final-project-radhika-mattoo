
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
#include <fstream>

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
VertexBufferObject VBO_CAT;
VertexBufferObject VBO_NUT;
int NUT_VERTICES = 8544;
int CAT_VERTICES = 12000;
int TOTAL = NUT_VERTICES + CAT_VERTICES;
// int TOTAL = ROSE_VERTICES;
MatrixXf V_CAT(3,CAT_VERTICES);
MatrixXf V_NUT(3,NUT_VERTICES);
//----------------------------------
// EDGES
//----------------------------------
VertexBufferObject VBO_OU;
VertexBufferObject VBO_OV;
MatrixXf OU(3, NUT_VERTICES);
MatrixXf OV(3, NUT_VERTICES);


//----------------------------------
// NORMALS
//----------------------------------
VertexBufferObject VBO_N_CAT;
VertexBufferObject VBO_N_NUT;
MatrixXf N_CAT(3,CAT_VERTICES);
MatrixXf N_NUT(3,NUT_VERTICES);
//----------------------------------
// TEXTURE
//----------------------------------
VertexBufferObject VBO_T_CAT;
VertexBufferObject VBO_T_NUT;
MatrixXf T_CAT(2,CAT_VERTICES);
MatrixXf T_NUT(2,NUT_VERTICES);
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

Vector3f lightPos(0.0, 1.0, focal_length);
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

MatrixXf dna_faces_ou = MatrixXf::Zero(3,NUT_VERTICES);
MatrixXf dna_vertices_ou = MatrixXf::Zero(3,NUT_VERTICES);

MatrixXf dna_faces_ov = MatrixXf::Zero(3,NUT_VERTICES);
MatrixXf dna_vertices_ov = MatrixXf::Zero(3,NUT_VERTICES);

vector< Vector3f > dna_out_vertices;
vector< Vector2f > dna_out_uvs;
vector< Vector3f > dna_out_normals;

vector< Vector3f > cat_out_vertices;
vector< Vector2f > cat_out_uvs;
vector< Vector3f > cat_out_normals;


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
void readObjFile(string filename, bool cat)
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

    if(cat){
      cat_out_vertices.push_back(vertex);
      cat_out_uvs.push_back(uv);
      cat_out_normals.push_back(normal);
    }else{
      dna_out_vertices.push_back(vertex);
      dna_out_uvs.push_back(uv);
      dna_out_normals.push_back(normal);
    }
  }

}

Vector2f calculateBarycenter(int column)
{
  // Get uv barycenter
  float coord_1_x = dna_out_uvs[column](0);
  float coord_1_y = dna_out_uvs[column](1);

  float coord_2_x = dna_out_uvs[column + 1](0);
  float coord_2_y = dna_out_uvs[column + 1](1);

  float coord_3_x = dna_out_uvs[column + 2](0);
  float coord_3_y = dna_out_uvs[column + 2](1);

  // Calculate barycenter
  float barycenter_x = (coord_1_x + coord_2_x + coord_3_x) / 3;
  float barycenter_y = (coord_1_y + coord_2_y + coord_3_y) / 3;

  Vector2f barycenter(barycenter_x, barycenter_y);

  return barycenter;
}

bool isInVector(vector<int> summed, int idx)
{
  for(int i = 0; i < summed.size(); i++){
    if(idx == summed[i]) return true;
  }
  return false;
}

// Calculates alpha, beta, gamma
Vector3f getParams(Vector2f p, float coord1_x, float coord1_y, float coord2_x, float coord2_y, float coord3_x, float coord3_y)
{
  Matrix3f A_;
  Vector3f b_;
  A_ << coord1_x, coord2_x, coord3_x, coord1_y, coord2_y, coord3_y, 1, 1, 1;
  b_ << p(0), p(1), 1;

  Vector3f sol = A_.colPivHouseholderQr().solve(b_);
  return sol;
  // float alpha = sol[0];
  // float beta = sol[1];
  // float gamma = sol[2];



}

void calculateDerivatives()
{
  float epsilon = 0.1;
  for(int i = 0; i < dna_out_uvs.size(); i+=3)
  {
    int column = i;
    float coord_1_x = dna_out_uvs[column](0);
    float coord_1_y = dna_out_uvs[column](1);

    float coord_2_x = dna_out_uvs[column + 1](0);
    float coord_2_y = dna_out_uvs[column + 1](1);

    float coord_3_x = dna_out_uvs[column + 2](0);
    float coord_3_y = dna_out_uvs[column + 2](1);

    // Get the 3 points
    Vector2f barycenter = calculateBarycenter(i);
    Vector2f p_u = barycenter + Vector2f(epsilon, 0.0);
    Vector2f p_v = barycenter + Vector2f(0.0, epsilon);


    // Get alpha, beta, gamma for the 3 2D points
    Vector3f p_params = getParams(barycenter, coord_1_x, coord_1_y,  coord_2_x, coord_2_y,  coord_3_x, coord_3_y);
    Vector3f p_u_params = getParams(p_u, coord_1_x, coord_1_y,  coord_2_x, coord_2_y,  coord_3_x, coord_3_y);
    Vector3f p_v_params = getParams(p_v, coord_1_x, coord_1_y,  coord_2_x, coord_2_y,  coord_3_x, coord_3_y);

    Vector3f V_p = (p_params(0) * V_NUT.col(i)) + (p_params(1) * V_NUT.col(i + 1)) + (p_params(2) * V_NUT.col(i + 2));
    Vector3f V_u = (p_u_params(0) * V_NUT.col(i)) + (p_u_params(1) * V_NUT.col(i + 1)) + (p_u_params(2) * V_NUT.col(i + 2));
    Vector3f V_v = (p_v_params(0) * V_NUT.col(i)) + (p_v_params(1) * V_NUT.col(i + 1)) + (p_v_params(2) * V_NUT.col(i + 2));
    Vector3f O_u = (V_u - V_p).normalized();
    Vector3f O_v = (V_v - V_p).normalized();

    // Save values
    dna_faces_ou.col(i) << O_u;
    dna_faces_ou.col(i + 1) << O_u;
    dna_faces_ou.col(i + 2) << O_u;
    dna_vertices_ou.col(i) << O_u;
    dna_vertices_ou.col(i + 1) << O_u;
    dna_vertices_ou.col(i + 2) << O_u;

    dna_faces_ov.col(i) << O_v;
    dna_faces_ov.col(i + 1) << O_v;
    dna_faces_ov.col(i + 2) << O_v;
    dna_vertices_ov.col(i) << O_v;
    dna_vertices_ov.col(i + 1) << O_v;
    dna_vertices_ov.col(i + 2) << O_v;
  }
  cout << "Done with faces, doing vertices" << endl;
  // Iterate through all 3d vertices and sum partial derivatives
  for(int i = 0; i < dna_out_vertices.size(); i++)
  {
    vector<int> summed_ov;
    vector<int> summed_ou;
    Vector3f current = dna_out_vertices[i];
    Vector3f sum_ou = dna_vertices_ou.col(i);
    Vector3f sum_ov = dna_vertices_ov.col(i);

    summed_ov.push_back(i);
    summed_ou.push_back(i);
    for(int j = 0; j < dna_out_vertices.size(); j++){
      if(i == j) continue;
      if(!isInVector(summed_ou, j)){
        Vector3f other = dna_out_vertices[j];
        if(other[0] == current[0] && other[1] == current[1] && other[2] == current[2] ){
          sum_ou += dna_vertices_ou.col(j);
          summed_ou.push_back(j);
        }
      }
      if(!isInVector(summed_ov, j)){
        Vector3f other = dna_out_vertices[j];
        if(other[0] == current[0] && other[1] == current[1] && other[2] == current[2]){
          sum_ov += dna_vertices_ov.col(j);
          summed_ov.push_back(j);
        }
      }
    }
    // normalize sums and insert into all vertex columns
    sum_ou = sum_ou.normalized();
    for(int k = 0; k < summed_ou.size(); k++){
      int idx = summed_ou[k];
      dna_vertices_ou.col(idx) = sum_ou;
    }
    sum_ov = sum_ov.normalized();
    for(int k = 0; k < summed_ov.size(); k++){
      int idx = summed_ov[k];
      dna_vertices_ov.col(idx) = sum_ov;
    }
    if((i % 1000) == 0){
      cout << "At index: " << i << "/8544" << endl;
    }

  } //end outer loop

  VBO_OU.update(dna_vertices_ou);
  VBO_OV.update(dna_vertices_ov);
}

void initialize(GLFWwindow* window)
{
  VertexArrayObject VAO;
  VAO.init();
  VAO.bind();

  // READ IN PARSED DATA
  VBO_CAT.init();
  VBO_NUT.init();
  VBO_N_CAT.init();
  VBO_N_NUT.init();
  VBO_T_CAT.init();
  VBO_T_NUT.init();
  VBO_OV.init();
  VBO_OU.init();

  // READ IN OBJ FILES
  filenames.push_back("../data/pear_obj/pear_tri.obj");
  filenames.push_back("../data/cat_obj/Cat.obj");
  readObjFile(filenames[0], false);
  readObjFile(filenames[1], true);

  cout << "Pear vertices: " << dna_out_vertices.size() << endl;
  cout << "Pear normals: " << dna_out_normals.size() << endl;
  cout << "Pear textures: " << dna_out_uvs.size() << endl;
  for(int i = 0; i < dna_out_vertices.size(); i++)
  {
    Vector3f v_data = dna_out_vertices[i];
    Vector3f n_data = dna_out_normals[i];
    Vector2f t_data = dna_out_uvs[i];
    v_data[0] -= 3;
    v_data[2] -= 0.5;
    V_NUT.col(i) << v_data[0], v_data[1], v_data[2];
    N_NUT.col(i) << n_data[0], n_data[1], n_data[2];
    T_NUT.col(i) << t_data[0], t_data[1];
  }
  calculateDerivatives();

  cout << "Cat vertices: " << cat_out_vertices.size() << endl;
  cout << "Cat normals: " << cat_out_normals.size() << endl;
  cout << "Cat textures: " << cat_out_uvs.size() << endl;

  for(int i = 0; i < cat_out_vertices.size(); i++)
  {
    Vector3f v_data = cat_out_vertices[i];
    Vector3f n_data = cat_out_normals[i];
    Vector2f t_data = cat_out_uvs[i];
    v_data /= 100;
    v_data[2] -= 2;
    V_CAT.col(i) << v_data[0], v_data[1], v_data[2];
    N_CAT.col(i) << n_data[0], n_data[1], n_data[2];
    T_CAT.col(i) << t_data[0], t_data[1];
  }

  VBO_CAT.update(V_CAT);
  VBO_NUT.update(V_NUT);
  VBO_N_NUT.update(N_NUT);
  VBO_N_CAT.update(N_CAT);
  VBO_T_NUT.update(T_NUT);
  VBO_T_CAT.update(T_CAT);
  // READ IN NORMALS
  //------------------------------------------
  // MODEL MATRIX
  //------------------------------------------
  model <<
  1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  // DNA
  float direction = (PI/180) * 90;
  MatrixXf rotation(4,4);
  // rotation <<
  // 1.,    0.,                  0.,                 0.,
  // 0.,    cos(direction),   sin(direction),  0.,
  // 0.,    -sin(direction),  cos(direction),  0.,
  // 0.,    0.,                  0.,                 1.;
  //
  // model.block(0,0,4,4) = model.block(0,0,4,4) * rotation;

  // CAT
  rotation <<
  cos(direction),  0.,  -sin(direction),   0.,
  0.,                 1.,  0.,                   0.,
  sin(direction),  0.,  cos(direction),    0.,
  0.,                 0.,  0.,                   1.;

  model.block(0,4,4,4) = model.block(0,4,4,4) * rotation;

  direction = (PI/180) * 15;
  rotation <<
  cos(direction),      sin(direction),  0.,  0.,
  -sin(direction),     cos(direction),  0.,  0.,
  0.,                     0.,                 1.,  0.,
  0.,                     0.,                 0.,  1.;
  model.block(0,4,4,4) = model.block(0,4,4,4) * rotation;

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
  float direction = (PI/180) * 20;
  MatrixXf rotation(4,4);
  if(action == GLFW_RELEASE){
    switch(key){
      // ROTATE CAT
      // rotate
      case GLFW_KEY_1:{
        rotation <<
          cos(direction),  0.,  -sin(direction),   0.,
          0.,                 1.,  0.,                   0.,
          sin(direction),  0.,  cos(direction),    0.,
          0.,                 0.,  0.,                   1.;
          model.block(0, 4, 4, 4) = model.block(0, 4, 4, 4) * rotation;
        break;
      }
      case GLFW_KEY_2:{
        direction = - direction;
        rotation <<
          cos(direction),  0.,  -sin(direction),   0.,
          0.,                 1.,  0.,                   0.,
          sin(direction),  0.,  cos(direction),    0.,
          0.,                 0.,  0.,                   1.;
          model.block(0, 4, 4, 4) = model.block(0, 4, 4, 4) * rotation;
        break;
      }
      case GLFW_KEY_3:{
        rotation <<
          1.,    0.,                  0.,                 0.,
          0.,    cos(direction),   sin(direction),  0.,
          0.,    -sin(direction),  cos(direction),  0.,
          0.,    0.,                  0.,                 1.;
          model.block(0, 4, 4, 4) = model.block(0, 4, 4, 4) * rotation;
        break;
      }
      case GLFW_KEY_4:{
        direction = -direction;
        rotation <<
          1.,    0.,                  0.,                 0.,
          0.,    cos(direction),   sin(direction),  0.,
          0.,    -sin(direction),  cos(direction),  0.,
          0.,    0.,                  0.,                 1.;
          model.block(0, 4, 4, 4) = model.block(0, 4, 4, 4) * rotation;
        break;
      }

      // NUT
      case GLFW_KEY_5:{
        rotation <<
          cos(direction),  0.,  -sin(direction),   0.,
          0.,                 1.,  0.,                   0.,
          sin(direction),  0.,  cos(direction),    0.,
          0.,                 0.,  0.,                   1.;
          model.block(0, 0, 4, 4) = model.block(0, 0, 4, 4) * rotation;
        break;
      }
      case GLFW_KEY_6:{
        direction = - direction;
        rotation <<
          cos(direction),  0.,  -sin(direction),   0.,
          0.,                 1.,  0.,                   0.,
          sin(direction),  0.,  cos(direction),    0.,
          0.,                 0.,  0.,                   1.;
          model.block(0, 0, 4, 4) = model.block(0, 0, 4, 4) * rotation;
        break;
      }
      case GLFW_KEY_7:{
        rotation <<
          1.,    0.,                  0.,                 0.,
          0.,    cos(direction),   sin(direction),  0.,
          0.,    -sin(direction),  cos(direction),  0.,
          0.,    0.,                  0.,                 1.;
          model.block(0, 0, 4, 4) = model.block(0, 0, 4, 4) * rotation;
        break;
      }
      case GLFW_KEY_8:{
        direction = -direction;
        rotation <<
          1.,    0.,                  0.,                 0.,
          0.,    cos(direction),   sin(direction),  0.,
          0.,    -sin(direction),  cos(direction),  0.,
          0.,    0.,                  0.,                 1.;
          model.block(0, 0, 4, 4) = model.block(0, 0, 4, 4) * rotation;
        break;
      }
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
                    "in vec3 ov_in;"
                    "in vec3 ou_in;"
                    "out vec2 TexCoord;" //UV coordinates
                    "out vec3 Normal;" // Normal
                    "out vec3 FragPos;" //Position
                    "out vec3 OU;"
                    "out vec3 OV;"
                    "uniform mat4 view;"
                    "uniform mat4 projection;"
                    "uniform mat4 model;"
                    "void main()"
                    "{"
                    "    gl_Position = projection * view * model * vec4(position, 1.0);"
                    "    FragPos = vec3(model * vec4(position, 1.0f));"
                    "    Normal =  normalize(mat3(transpose(inverse(model) )) * normal);"
                    "    TexCoord = texCoord;"
                    "    OV = normalize(mat3(transpose(inverse(model))) * ov_in);"
                    "    OU = normalize(mat3(transpose(inverse(model))) * ou_in);"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "out vec4 outColor;"
                    "in vec2 TexCoord;"
                    "in vec3 Normal;"
                    "in vec3 FragPos;"
                    "in vec3 OU;"
                    "in vec3 OV;"
                    "uniform vec3 viewPos;"
                    "uniform vec3 lightPos;"
                    "uniform sampler2D ourTexture;"
                    "uniform sampler2D bump;"
                    "uniform bool is_cat;"
                    "uniform float OFF;"
                    "void main()"
                    "{"
                  "       float ambientStrength = 0.05f;"
                  "       vec3 lightColor = vec3(1.0, 1.0, 1.0);"
                  "       float lightPower = 10.0;"
                  "       if(is_cat){" //CAT
                  "           vec3 norm = normalize(Normal);"
                  "           vec3 lightDir = normalize(lightPos - FragPos);"
                  "           vec3 viewDir = normalize(viewPos - FragPos);"
                  "           vec3 reflectDir = reflect(-lightDir, norm);  "
                              //AMBIENT
                  "           vec3 ambient = ambientStrength * lightColor;"
                              //DIFFUSE
                  "           float diff = max(dot(norm, lightDir), 0);"
                  "           vec3 diffuse = diff * lightColor;"
                              //SPECULAR
                  "           float specularStrength = 0.5f;"
                  "           float spec = pow(max(dot(viewDir, reflectDir), 0.0), 100);"
                  "           vec3 specular = specularStrength * spec * lightColor;  "
                              // TOTAL
                  "           vec3 result = (ambient + diffuse + specular);"
                  "           outColor =  texture(ourTexture, TexCoord) * vec4(result, 1.0);"
                  "       }else{"
                  "           float Bu = dFdx(texture(bump, TexCoord).x);"
                  "           float Bv = dFdx(texture(bump, TexCoord).y);"
                  "           vec3 A = cross(Normal, OV);"
                  "           vec3 B = cross(Normal, OU);"
                  "           vec3 D = Bu*A - Bv*B;"
                  "           vec3 norm = normalize(Normal + D);"

                  "           vec3 lightDir = normalize(lightPos - FragPos);"
                  "           vec3 viewDir = normalize(viewPos - FragPos);"
                  "           vec3 reflectDir = reflect(-lightDir, norm);  "
                              //AMBIENT
                  "           vec3 ambient = ambientStrength * lightColor ;"
                              //DIFFUSE
                  "           float diff = max(dot(norm, lightDir), 0);"
                  "           vec3 diffuse = diff * lightColor;"
                              //SPECULAR
                  "           float specularStrength = 0.5f;"
                  "           float spec = pow(max(dot(viewDir, reflectDir), 0.0), 96);"
                  "           vec3 specular = specularStrength * spec * lightColor * vec3(0.500000, 0.500000 ,0.500000);;  "
                              // TOTAL
                  "           vec3 result = (ambient + diffuse + specular);"

                  "           outColor = texture(ourTexture, TexCoord) * vec4(result, 1.0);"

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

    program.bindVertexAttribArray("ov_in", VBO_OV);
    program.bindVertexAttribArray("ou_in", VBO_OU);

    // UNIFORMS
    glUniform3f(program.uniform("lightPos"), lightPos[0] ,lightPos[1], lightPos[2]);
    glUniform3f(program.uniform("viewPos"), eye[0], eye[1], eye[2]);
    glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(program.uniform("projection"), 1, GL_FALSE, projection.data());


    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();


    unsigned int texture1, texture2, texture3;
    // -------------------------
    // LOAD NUT TEXTURE
    // -------------------------
    // texture 1
    // ---------
    cout << "Loading Pear texture/bump maps" << endl;

    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1); // All upcoming GL_TEXTURE_2D operations now have effect on our texture object
    // Set our texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Set texture filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
     int width, height, nrChannels;
     stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
     // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
     unsigned char *data = stbi_load("../data/pear_obj/pear_texture.jpg", &width, &height, &nrChannels, 0);
     if (data)
     {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
     }
     else
     {
        std::cout << "Failed to load Pear texture" << std::endl;
     }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done, so we won't accidentily mess up our texture.



    glGenTextures(1, &texture3);
    glBindTexture(GL_TEXTURE_2D, texture3); // All upcoming GL_TEXTURE_2D operations now have effect on our texture object
    // Set our texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT
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
     data  = stbi_load("../data/hazelnut_obj/carpet_noise.jpeg", &width, &height, &nrChannels, 0);
     if (data)
     {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
     }
     else
     {
        std::cout << "Failed to load Pear bump map" << std::endl;
     }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 2); // Unbind texture when done, so we won't accidentily mess up our texture.



    // -------------------------
    // LOAD CAT TEXTURE
    // -------------------------
    // texture 2
    // ---------
    cout << "Loading Cat texture" << endl;
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
    data = stbi_load("../data/cat_obj/Cat_texture.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
       glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
       glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
       std::cout << "Failed to load Cat texture" << std::endl;
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
      // NUT
      //--------
      program.bindVertexAttribArray("position",VBO_NUT);
      program.bindVertexAttribArray("normal",VBO_N_NUT);
      program.bindVertexAttribArray("texCoord",VBO_T_NUT);

      glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.block(0,0,4,4).data());
      glUniform1i(program.uniform("is_cat"), false);
      glUniform1i(program.uniform("ourTexture"), 0);
      glUniform1i(program.uniform("bump"), 2);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texture1); //texture
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, texture3); //height map
      for(int i = 0; i < dna_out_vertices.size(); i+=3){
        glDrawArrays(GL_TRIANGLES, i , 3);
      }

      //--------
      // CAT
      //--------
      program.bindVertexAttribArray("position",VBO_CAT);
      program.bindVertexAttribArray("normal",VBO_N_CAT);
      program.bindVertexAttribArray("texCoord",VBO_T_CAT);
      glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.block(0,4,4,4).data());
      glUniform1i(program.uniform("is_cat"), true);
      glUniform1i(program.uniform("ourTexture"), 1);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, texture2); // cat texture
      for(int i = 0; i < cat_out_vertices.size(); i+=3){
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
