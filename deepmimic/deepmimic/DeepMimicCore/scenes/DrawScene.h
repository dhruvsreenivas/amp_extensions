#pragma once

#include "Scene.h"
#include "render/Camera.h"
#include "render/ShadowMap.h"
#include "render/Shader.h"

class cDrawScene : virtual public cScene
{
public:

	enum eCamTrackMode
	{
		eCamTrackModeXZ,
		eCamTrackModeY,
		eCamTrackModeXYZ,
		eCamTrackModeStill,
		eCamTrackModeFixed,
		eCamTrackModeMax
	};

	virtual ~cDrawScene();
	
	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();
	virtual void Reset();
	virtual void ResetIndex(int index, bool resolve = true, bool noise_bef_rot = false, 
		double min = 0, double max = 0, double radian = 0, bool rot_vel_w_pose = false, 
		bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual void ResetTime(double time, bool resolve = true, bool noise_bef_rot = false, 
		double min = 0, double max = 0, double radian = 0, bool rot_vel_w_pose = false, 
		bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	
	virtual void Draw();

	virtual void MouseClick(int button, int state, double device_x, double device_y);
	virtual void MouseMove(double device_x, double device_y);

	virtual void EnableDrawInfo(bool enable);

	virtual std::string GetName() const;

protected:
	eCamTrackMode mCamTrackMode;

	cCamera mCamera;
	cCamera mShadowCam;

	bool mDrawInfo;

	// textures
	std::unique_ptr<cShadowMap> mShadowMap;
	std::unique_ptr<cTextureDesc> mGridTex;

	// shaders
	std::string mDeepMimicPath; //new. 
	/*This should be the path to the DeepMimic(not deepmimiccore) directory.
	* This path will depend on the entry point of the program creating the DeepMimicCore. Default is "".
	*/
	std::unique_ptr<cShader> mShaderMesh;
	std::unique_ptr<cShader> mShaderDepth;

	// shader uniform handles
	GLuint mMeshLightDirHandle;
	GLuint mMeshLightColourHandle;
	GLuint mMeshAmbientColourHandle;
	GLuint mMeshShadowProjHandle;
	GLuint mMeshMaterialDataHandle;
	GLuint mMeshFogColorHandle;
	GLuint mMeshFogDataHandle;

	cDrawScene();
	virtual void ParseCamTrackMode(const std::shared_ptr<cArgParser>& parser, eCamTrackMode& out_mode) const;
	virtual void Reshape(int w, int h);

	virtual void ResizeCamera(int w, int h);
	virtual void InitCamera();

	virtual eCamTrackMode GetCamTrackMode() const;
	virtual void UpdateCamera();
	virtual void UpdateCameraTracking();
	virtual void UpdateCameraStill();
	virtual double GetCamStillSnapDistX() const;

	virtual tVector GetCamTrackPos() const;
	virtual tVector GetCamStillPos() const;
	virtual void ResetCamera();
	virtual tVector GetDefaultCamFocus() const;

	virtual void ClearFrame();
	virtual tVector GetBkgColor() const;

	virtual void SetupView();
	virtual void RestoreView();
	virtual tVector GetLightDirection() const;

	virtual void DrawScene();
	virtual void DrawGrid() const;
	virtual void DrawGroundMainScene();
	virtual void DrawCharacterMainScene();
	virtual void DrawObjsMainScene();
	virtual void DrawMiscMainScene();
	virtual void DrawGround() const;
	virtual void DrawCharacters() const;
	virtual void DrawObjs() const;
	virtual void DrawMisc() const;
	virtual void DrawInfo() const;

	virtual tVector GetVisOffset() const;
	virtual tVector GetLineColor() const;
	virtual tVector GetGroundColor() const;

	virtual void InitRenderResources();
	virtual bool LoadTextures(); //new, modified
	virtual void SetupMeshShader();
	virtual void DoShadowPass();
};