#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "util/MathUtil.h"
#include "util/ArgParser.h"
#include "util/Timer.h"

class cScene
{
public:
	virtual ~cScene();
	
	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();
	virtual void Clear();
	virtual void Reset();
	virtual void ResetIndex(int index, bool resolve = true, bool noise_bef_rot = false, 
		double min = 0, double max = 0, double radian = 0, bool rot_vel_w_pose = false, 
		bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual void ResetTime(double time, bool resolve = true, bool noise_bef_rot = false, 
		double min = 0, double max = 0, double radian = 0, bool rot_vel_w_pose = false, 
		bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual void Update(double timestep);
	virtual void UpdateSyncTimestep(double timestep);//new

	virtual void Draw();
	virtual void Keyboard(unsigned char key, double device_x, double device_y);
	virtual void MouseClick(int button, int state, double device_x, double device_y);
	virtual void MouseMove(double device_x, double device_y);
	virtual void Reshape(int w, int h);

	virtual void Shutdown();
	virtual bool IsDone() const;
	virtual double GetTime() const;

	virtual bool HasRandSeed() const;
	virtual void SetRandSeed(unsigned long seed);
	virtual unsigned long GetRandSeed() const;

	virtual bool IsEpisodeEnd() const;
	virtual bool CheckValidEpisode() const;

	virtual std::string GetName() const = 0;
	virtual double GetMotionLength() const ; //new
	virtual double GetAgentUpdateRate(int agent_id) const; //new
	virtual int GetStatePoseOffset(int agent_id) const; //new 
	virtual int GetStatePoseSize(int agent_id) const; //new 
	virtual int GetStateVelOffset(int agent_id) const; //new 
	virtual int GetStateVelSize(int agent_id) const; //new
	virtual int GetStatePhaseOffset(int agent_id) const; //new 
	virtual int GetStatePhaseSize(int agent_id) const; //new

	virtual int GetPosFeatureDim(int agent_id) const;// new
	virtual int GetRotFeatureDim(int agent_id) const;// new

protected:

	cRand mRand;
	bool mHasRandSeed;
	unsigned long mRandSeed;

	std::shared_ptr<cArgParser> mArgParser;
	cTimer::tParams mTimerParams;
	cTimer mTimer;

	cScene();

	virtual void ResetParams();
	virtual void ResetScene();
	virtual void ResetSceneIndex(int index, bool resolve = true, bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0,
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual void ResetSceneTime(double time, bool resolve = true, bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0,
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false);//new
	virtual void InitTimers();
	virtual void ResetTimers();
	virtual void UpdateTimers(double timestep);
};