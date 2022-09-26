#include "Scene.h"

cScene::cScene()
{
	mRand.Seed(cMathUtil::RandUint());
	mRandSeed = 0;
	mHasRandSeed = false;
}

cScene::~cScene()
{
}

void cScene::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	mArgParser = parser;

	std::string timer_type_str = "";
	mArgParser->ParseString("timer_type", timer_type_str);
	mTimerParams.mType = cTimer::ParseTypeStr(timer_type_str);
	mArgParser->ParseDouble("time_lim_min", mTimerParams.mTimeMin);
	mArgParser->ParseDouble("time_lim_max", mTimerParams.mTimeMax);
	mArgParser->ParseDouble("time_lim_exp", mTimerParams.mTimeExp);
}

void cScene::Init()
{
	if (HasRandSeed())
	{
		SetRandSeed(mRandSeed);
	}

	InitTimers();
	ResetParams();
}

void cScene::Clear()
{
	ResetParams();
}

void cScene::Reset()
{
	ResetScene();
}

void cScene::ResetIndex(int index, bool resolve, bool noise_bef_rot, double min, double max, double radian, bool rot_vel_w_pose,
	bool vel_noise, double interp, bool knee_rot)
{

	ResetSceneIndex(index, resolve, noise_bef_rot, min, max, radian, rot_vel_w_pose, vel_noise, interp, knee_rot);
}

void cScene::ResetTime(double time, bool resolve, bool noise_bef_rot, double min, double max, double radian, bool rot_vel_w_pose,
	bool vel_noise, double interp, bool knee_rot)
{

	ResetSceneTime(time, resolve, noise_bef_rot, min, max, radian, rot_vel_w_pose, vel_noise, interp, knee_rot);
}

void cScene::Update(double timestep)
{
	UpdateTimers(timestep);
}

void cScene::UpdateSyncTimestep(double timestep)
{
	UpdateTimers(timestep);
}

void cScene::Draw()
{
}

void cScene::Keyboard(unsigned char key, double device_x, double device_y)
{
}

void cScene::MouseClick(int button, int state, double device_x, double device_y)
{
}

void cScene::MouseMove(double device_x, double device_y)
{
}


void cScene::Reshape(int w, int h)
{
}

void cScene::Shutdown()
{
}

bool cScene::IsDone() const
{
	return false;
}

double cScene::GetTime() const
{
	return mTimer.GetTime();
}

bool cScene::HasRandSeed() const
{
	return mHasRandSeed;
}

void cScene::SetRandSeed(unsigned long seed)
{
	mHasRandSeed = true;
	mRandSeed = seed;
	mRand.Seed(seed);
}

unsigned long cScene::GetRandSeed() const
{
	return mRandSeed;
}

bool cScene::IsEpisodeEnd() const
{
	return mTimer.IsEnd();
}

bool cScene::CheckValidEpisode() const
{
	return true;
}

double cScene::GetMotionLength() const
{
	return 0.0;
}

double cScene::GetAgentUpdateRate(int agent_id) const
{
	return 0.0;
}

int cScene::GetStatePoseOffset(int agent_id) const
{
	return 0;
}

int cScene::GetStatePoseSize(int agent_id) const
{
	return 0;
}

int cScene::GetStateVelOffset(int agent_id) const
{
	return 0;
}

int cScene::GetStateVelSize(int agent_id) const
{
	return 0;
}

int cScene::GetStatePhaseOffset(int agent_id) const
{
	return 0;
}

int cScene::GetStatePhaseSize(int agent_id) const
{
	return 0;
}

int cScene::GetPosFeatureDim(int agent_id) const
{
	return 0;
}

int cScene::GetRotFeatureDim(int agent_id) const
{
	return 0;
}


void cScene::ResetParams()
{
	ResetTimers();
}

void cScene::ResetScene()
{
	ResetParams();
}

void cScene::ResetSceneIndex(int index, bool resolve, bool noise_bef_rot, double min, double max, double radian, bool rot_vel_w_pose,
	bool vel_noise, double interp, bool knee_rot)
{
	ResetParams();
}

void cScene::ResetSceneTime(double time, bool resolve, bool noise_bef_rot, double min, double max, double radian, bool rot_vel_w_pose,
	bool vel_noise, double interp, bool knee_rot)
{
	ResetParams();
}


void cScene::InitTimers()
{
	mTimer.Init(mTimerParams);
}

void cScene::ResetTimers()
{
	mTimer.Reset();
}

void cScene::UpdateTimers(double timestep)
{
	mTimer.Update(timestep);
}