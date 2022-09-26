#pragma once

#include "scenes/RLSceneSimChar.h"
#include "anim/KinCharacter.h"
#include "anim/KinCtrlBuilder.h"

class cSceneImitate : virtual public cRLSceneSimChar
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cSceneImitate();
	virtual ~cSceneImitate();

	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();

	virtual const std::shared_ptr<cKinCharacter>& GetKinChar() const;
	virtual void EnableRandRotReset(bool enable);
	virtual bool EnabledRandRotReset() const;

	virtual double CalcReward(int agent_id) const;
	virtual eTerminate CheckTerminate(int agent_id) const;
	virtual double GetMotionLength() const; //new

	virtual std::string GetName() const;
	

	virtual double GetAgentUpdateRate(int agent_id) const;//new
	virtual int GetStatePoseOffset(int agent_id) const; //new 
	virtual int GetStatePoseSize(int agent_id) const; //new 
	virtual int GetStateVelOffset(int agent_id) const; //new 
	virtual int GetStateVelSize(int agent_id) const; //new
	virtual int GetStatePhaseOffset(int agent_id) const; //new 
	virtual int GetStatePhaseSize(int agent_id) const; //new
	virtual int GetPosFeatureDim(int agent_id) const;//new
	virtual int GetRotFeatureDim(int agent_id) const;//new

protected:

	cKinCtrlBuilder::tCtrlParams mKinCtrlParams;
	std::shared_ptr<cKinCharacter> mKinChar;

	Eigen::VectorXd mJointWeights;
	bool mEnableRandRotReset;
	bool mSyncCharRootPos;
	bool mSyncCharRootRot;
	bool mEnableRootRotFail;

	virtual void ParseKinCtrlParams(const std::shared_ptr<cArgParser>& parser, cKinCtrlBuilder::tCtrlParams& out_params) const;
	virtual bool BuildCharacters();

	virtual void CalcJointWeights(const std::shared_ptr<cSimCharacter>& character, Eigen::VectorXd& out_weights) const;
	virtual bool BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl);
	virtual bool BuildKinCharacter();
	virtual bool BuildKinController();
	virtual void UpdateCharacters(double timestep);
	virtual void UpdateCharactersSyncTimestep(double timestep); //New
	virtual void UpdateKinChar(double timestep);

	virtual void ResetCharacters();
	virtual void ResetKinChar();
	
	virtual void ResetCharactersTime(double time, bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0, 
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false); //New
	virtual void ResetCharactersIndex(int index, bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0, 
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false); //New
	virtual void ResetKinCharTime(double time, bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0, 
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false);//New
	virtual void ResetKinCharIndex(int index,  bool noise_bef_rot = false, double min = 0, double max = 0, double radian = 0, 
		bool rot_vel_w_pose = false, bool vel_noise = false, double interp = 1, bool knee_rot = false); //New
	virtual void SyncCharacters();
	virtual bool EnableSyncChar() const;
	virtual void InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char);

	virtual void InitJointWeights();
	virtual void ResolveCharGroundIntersect();
	virtual void ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const;
	virtual void SyncKinCharRoot();
	virtual void SyncKinCharNewCycle(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const;

	virtual double GetKinTime() const;
	virtual bool CheckKinNewCycle(double timestep) const;
	virtual bool HasFallen(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const;
	
	virtual double CalcRandKinResetTime();
	virtual double CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const;
};