#pragma once

#include "scenes/DrawScene.h"
#include "scenes/SceneSimChar.h"
#include "sim/ObjTracer.h"

#include "render/DrawMesh.h"

class cDrawSceneSimChar : virtual public cDrawScene
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	cDrawSceneSimChar();
	virtual ~cDrawSceneSimChar();

	virtual void Init();
	virtual void Clear();
	virtual void Update(double time_elapsed);
	virtual void UpdateSyncTimestep(double time_elapsed);//New

	virtual const std::shared_ptr<cSceneSimChar>& GetScene() const;
	
	virtual void MouseClick(int button, int state, double x, double y);
	virtual void MouseMove(double x, double y);
	virtual void Keyboard(unsigned char key, double device_x, double device_y);
	
	virtual double GetTime() const;
	virtual bool IsEpisodeEnd() const;
	virtual bool CheckValidEpisode() const;

	virtual std::string GetName() const;

protected:

	std::shared_ptr<cSceneSimChar> mScene;

	std::unique_ptr<cDrawMesh> mGroundDrawMesh;
	size_t mPrevGroundUpdateCount;

	bool mEnableTrace;
	cObjTracer mTracer;
	std::vector<int> mTraceHandles;

	// UI stuff
	tVector mClickScreenPos;
	tVector mDragScreenPos;
	tVector mSelectObjLocalPos;
	cSimObj* mSelectedObj;

	int mTracerBufferSize;
	double mTracerSamplePeriod;

	virtual void BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const;
	virtual void SetupScene(std::shared_ptr<cSceneSimChar>& out_scene);
	virtual void UpdateScene(double time_elapsed);
	virtual void UpdateSceneSyncTimestep(double time_elapsed); //new
	virtual void ResetScene();
	virtual void ResetSceneIndex(int index, bool resolve = true,  bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0,
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual void ResetSceneTime(double time, bool resolve = true, bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0,
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual double GetMotionLength() const; //new
	virtual double GetAgentUpdateRate(int agent_id) const; //new
	virtual int GetStatePoseOffset(int agent_id) const; //new 
	virtual int GetStatePoseSize(int agent_id) const; //new 
	virtual int GetStateVelOffset(int agent_id) const; //new 
	virtual int GetStateVelSize(int agent_id) const; //new
	virtual int GetStatePhaseOffset(int agent_id) const; //new 
	virtual int GetStatePhaseSize(int agent_id) const; //new
	virtual tVector GetCamTrackPos() const;
	virtual tVector GetCamStillPos() const;
	virtual tVector GetDefaultCamFocus() const;
	
	virtual void ResetParams();

	virtual void ToggleTrace();
	virtual void InitTracer();
	virtual void AddTraces();
	virtual void AddCharTrace(const std::shared_ptr<cSimCharacter>& character,
								const tVectorArr& cols);
	virtual void UpdateTracer(double time_elapsed);

	virtual void SpawnProjectile();
	virtual void SpawnBigProjectile();

	virtual void OutputCharState(const std::string& out_file) const;
	virtual std::string GetOutputCharFile() const;

	virtual void ResetUI();
	virtual void RayTest(const tVector& start, const tVector& end, cWorld::tRayTestResult& out_result);
	virtual bool ObjectSelected() const;
	virtual void HandleRayTest(const cWorld::tRayTestResult& result);
	virtual void ApplyUIForce(double time_step);

	virtual void DrawObjs() const;
	virtual void DrawObj(int obj_id) const;
	virtual void DrawMisc() const;

	virtual void DrawCoM() const;
	virtual void DrawTorque() const;
	virtual void DrawBodyVel() const;
	virtual void DrawHeading() const;
	virtual void DrawTrace() const;

	virtual void DrawPerturbs() const;

	virtual void DrawGround() const;
	virtual void DrawCharacters() const;
	virtual void DrawCharacter(const std::shared_ptr<cSimCharacter>& character) const;
	virtual void UpdateGroundDrawMesh();
	virtual void BuildGroundDrawMesh();

	virtual void DrawInfo() const;
	virtual void DrawPoliInfo() const;
};
