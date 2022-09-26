BIN=Main
OBJ_DIR = objs
MKDIR = mkdir -p

EIGEN_DIR =  <path to conda env>/include/eigen3
BULLET_INC_DIR = <path to conda env>/include/bullet

PYTHON_INC = <path to conda env>/include/python3.7m
PYTHON_LIB = <path to conda env>/lib/python3.7

GL_INC = <path to conda env>/include
GL_LIB = <path to conda env>/lib

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR)\
	-I$(GL_INC)

LIBS = -lGLEW -lGLU -lglut -lBulletDynamics -lBulletCollision -lLinearMath -lm -lstdc++ -lGL 

CC = clang++
CFLAGS = -std=c++11 -O3 -Wall -fPIC 

SRC_DIRS = util/ \
	util/json/ \
	anim/ \
	sim/ \
	render/ \
	render/lodepng/ \
	scenes/ \
	scenes/arm/ \
	scenes/particle/ \
	scenes/pendulum/

SRC_FILES = $(BIN).cpp DeepMimicCore.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp))
OUT_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_DIRS))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRC_FILES)))


default: build

build: $(BIN)

print:
	$(OUT_DIRS)

python: $(OUT_DIRS) $(OBJ)
	swig -c++ -python DeepMimicCore.i
	$(CC) -c -g $(CFLAGS) $(INC) -L$(GL_LIB) $(LIBS) DeepMimicCore.cpp DeepMimicCore_wrap.cxx -I$(PYTHON_INC) 
	$(CC) -shared $(CFLAGS) $(OBJ) -L$(GL_LIB) $(LIBS) DeepMimicCore_wrap.o -o _DeepMimicCore.so -L$(PYTHON_LIB) 

$(OBJ_DIR)/%.o: %.cpp
	$(CC) -c -g $(CFLAGS) $(INC) -o $@ $<

$(BIN): $(OUT_DIRS) $(OBJ)
	$(CC) -g $(CFLAGS) $(OBJ) $(LIBS) -o $@

$(OUT_DIRS):
	$(MKDIR) $(OUT_DIRS)
	
clean:
	- rm -f -r $(OUT_DIRS) $(BIN) $(OBJ) $(OBJ_DIR) DeepMimicCore_wrap.cxx _DeepMimicCore.so
