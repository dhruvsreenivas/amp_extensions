#include "KinCharacter.h"
#include <assert.h>
#include <functional>
#include "util/FileUtil.h"
#include "anim/MotionController.h"
const double gDiffTimeStep = 1 / 600.0;

cKinCharacter::tParams::tParams()
{
	mID = gInvalidIdx;
	mCharFile = "";
	mOrigin.setZero();
	mLoadDrawShapes = true;
}

cKinCharacter::cKinCharacter()
{
	mOrigin.setZero();
	mOriginRot.setIdentity();
}

cKinCharacter::~cKinCharacter()
{
}

bool cKinCharacter::Init(const tParams& params)
{
	mID = params.mID;
	mOrigin.setZero();

	bool succ = cCharacter::Init(params.mCharFile, params.mLoadDrawShapes);
	if (succ)
	{
		if (params.mStateFile != "")
		{
			bool succ_state = ReadState(params.mStateFile);

			if (!succ_state)
			{
				printf("Failed to load character state from %s\n", params.mStateFile.c_str());
			}
			else
			{
				mPose0 = mPose;
				mVel0 = mVel;
				SetPose(mPose);
				SetVel(mVel);
			}
		}

		SetOriginPos(params.mOrigin);
	}
	else
	{
		printf("Failed to build character from char_file: %s\n", params.mCharFile.c_str());
	}

	return succ;
}

void cKinCharacter::Clear()
{
	cCharacter::Clear();

	if (HasController())
	{
		mController->Clear();
	}
}

void cKinCharacter::Update(double time_step)
{
	cCharacter::Update(time_step);

	if (HasController())
	{
		mController->Update(time_step);
	}
	
	Pose();
}

void cKinCharacter::Reset()
{
	cCharacter::Reset();
	if (HasController())
	{
		mController->Reset();
	}
	Pose();
}

bool cKinCharacter::LoadMotion(const std::string& motion_file)
{
	bool succ = true;
	std::shared_ptr<cMotionController> ctrl = std::shared_ptr<cMotionController>(new cMotionController());
	ctrl->Init(this, motion_file);
	SetController(ctrl);
	return succ;
}

const cMotion* cKinCharacter::GetMotion() const
{
	const cMotion* motion = nullptr;
	if (HasController())
	{
		motion = &(mController->GetMotion());
	}
	return motion;
}
 
cMotion* cKinCharacter::GetMotion()
{
	cMotion* motion = nullptr;
	if (HasController())
	{
		motion = &(mController->GetMotion());
	}
	return motion;
}

double cKinCharacter::GetMotionDuration() const
{
	double dur = 0;
	if (HasController())
	{
		dur = mController->GetMotionDuration();
	}
	return dur;
}

void cKinCharacter::ChangeMotionDuration(double dur)
{
	if (HasController())
	{
		mController->ChangeMotionDuration(dur);
	}
}

bool cKinCharacter::EnableMotionLoop() const
{
	bool loop = false;
	if (HasController())
	{
		loop = mController->EnableMotionLoop();
	}
	return loop;
}

int cKinCharacter::GetNumMotionFrames() const
{
	int num_frames = 0;
	if (HasController())
	{
		num_frames = mController->GetNumMotionFrames();
	}
	return num_frames;
}

void cKinCharacter::SetTime(double time)
{
	if (HasController())
	{
		mController->SetTime(time);
	}
}

void cKinCharacter::SetTimeFromIndex(int index)
{
	if (HasController())
	{
		mController->SetTimeFromIndex(index);
	}
}

double cKinCharacter::GetTime() const
{
	double time = 0;
	if (HasController())
	{
		time = mController->GetTime();
	}
	return time;
}

int cKinCharacter::GetCycle() const
{
	int cycle = 0;
	if (HasController())
	{
		cycle = mController->GetCycle();
	}
	return cycle;
}

double cKinCharacter::GetPhase() const
{
	double phase = 0;
	if (HasController())
	{
		phase = mController->GetPhase();
	}
	return phase;
}

void cKinCharacter::Pose()
{
	Pose(GetTime());
}

void cKinCharacter::Pose(double time)
{
	CalcPose(time, mPose);
	SetPose(mPose);

	CalcVel(time, mVel);
	SetVel(mVel);
}

void cKinCharacter::PoseIndex(int index)
{
	CalcIndexPose(index, mPose);
	SetPose(mPose);

	CalcIndexVel(index, mVel);
	SetVel(mVel);
}

void cKinCharacter::BuildAcc(Eigen::VectorXd& out_acc) const
{
	CalcAcc(GetTime(), out_acc);
}

cMotion::tFrame cKinCharacter::GetMotionFrame(int f) const
{
	if (HasController())
	{
		return mController->GetMotionFrame(f);
	}
	return mPose;
}

cMotion::tFrame cKinCharacter::GetMotionFrameVel(int f) const
{
	if (HasController())
	{
		return mController->GetMotionFrameVel(f);
	}
	return mVel;
}

void cKinCharacter::SetRootPos(const tVector& pos)
{
	tVector root_pos = GetRootPos();
	tVector delta = pos - root_pos;
	MoveOrigin(delta);
}

void cKinCharacter::SetRootRotation(const tQuaternion& q)
{
	tQuaternion root_rot = GetRootRotation();
	tQuaternion dq = q * root_rot.conjugate();
	RotateOrigin(dq);
}

const tVector& cKinCharacter::GetOriginPos() const
{
	return mOrigin;
}

void cKinCharacter::SetOriginPos(const tVector& origin)
{
	tVector delta = origin - mOrigin;
	MoveOrigin(delta);
	mOrigin = origin; // this is needed in canse of NaNs
}

void cKinCharacter::MoveOrigin(const tVector& delta)
{
	mOrigin += delta;

	tVector root0 = cKinTree::GetRootPos(mPose0);
	root0 += delta;
	cKinTree::SetRootPos(root0, mPose0);

	tVector root = cKinTree::GetRootPos(mPose);
	root += delta;
	cKinTree::SetRootPos(root, mPose);
}

const tQuaternion& cKinCharacter::GetOriginRot() const
{
	return mOriginRot;
}

void cKinCharacter::SetOriginRot(const tQuaternion& rot)
{
	tQuaternion delta_rot = cMathUtil::QuatDiff(mOriginRot, rot);
	RotateOrigin(delta_rot);
	mOriginRot = rot; // this is needed in case of NaNs
}

void cKinCharacter::RotateOrigin(const tQuaternion& rot)
{
	mOriginRot = rot * mOriginRot;
	mOriginRot.normalize();

	tVector root_pos = GetRootPos();
	tVector root_pos_delta = mOrigin - root_pos;
	root_pos_delta = cMathUtil::QuatRotVec(rot, root_pos_delta);
	mOrigin = root_pos + root_pos_delta;

	tQuaternion root_rot0 = cKinTree::GetRootRot(mPose0);
	root_rot0 = rot * root_rot0;
	root_rot0.normalize();
	cKinTree::SetRootRot(root_rot0, mPose0);

	tQuaternion root_rot = cKinTree::GetRootRot(mPose);
	root_rot = rot * root_rot;
	root_rot.normalize();
	cKinTree::SetRootRot(root_rot, mPose);

	tVector vel0 = cKinTree::GetRootVel(mVel0);
	vel0 = cMathUtil::QuatRotVec(rot, vel0);
	cKinTree::SetRootVel(vel0, mVel0);

	tVector vel = cKinTree::GetRootVel(mVel);
	vel = cMathUtil::QuatRotVec(rot, vel);
	cKinTree::SetRootVel(vel, mVel);

	tVector ang_vel0 = cKinTree::GetRootAngVel(mVel0);
	ang_vel0 = cMathUtil::QuatRotVec(rot, ang_vel0);
	cKinTree::SetRootAngVel(ang_vel0, mVel0);

	tVector ang_vel = cKinTree::GetRootAngVel(mVel);
	ang_vel = cMathUtil::QuatRotVec(rot, ang_vel);
	cKinTree::SetRootAngVel(ang_vel, mVel);
}

void cKinCharacter::AddNoise(bool noise_bef_rot, double min, double max, double radian, bool rot_vel_w_pose, bool vel_noise, double interp, bool knee_rot)
{	

	if (noise_bef_rot) {
		AddNoisePoseVel(min, max);
		RandomRotatePoseVel(radian, rot_vel_w_pose, vel_noise, interp, knee_rot);
	}
	else {

		RandomRotatePoseVel(radian, rot_vel_w_pose, vel_noise, interp, knee_rot);
		AddNoisePoseVel(min, max);
	}

}

void cKinCharacter::AddNoisePoseVel(double min, double max) 
{
	/*
	Samples a noisy vector to add to the pose and velocity state. The vectors are sampled from 
	U(min, max)
	*/
	if (min == 0 && max == 0) {
		return;
	}
	mPose += cMathUtil::RandDoubleEigen(mPose.size(), min, max);
	mVel += cMathUtil::RandDoubleEigen(mVel.size(), min, max);
}
void cKinCharacter::RandomRotatePoseVel(double radian, bool rot_vel_w_pose, bool vel_noise, double interp, bool knee_rot) 
{
	if (radian == 0) {
		return;
	}
	/*
	Parameter radian : (float)When adding noise by sampling a random rotation to add to each joint, the angles are sampled from
						the uniform distribution U(-radian, radian).
	Parameter rot_vel_w_pose : (bool)If true, this will add the same noisy rotation to the angular velocity of the joint that is
								added to the joint's rotation.
	Parameter vel_noise : (bool)If true, we also sample random rotations to add to the angular velocities.
	Parameter interp : (float, precondition: should be in[0, 1]).If 1, the velocitgy is reset to the same as the
					the expert which is what AMP does.If 0, the velocity is set to 0. Otherwise, the velocity
					is set by interpolating between 0 and the expert based on the interp value.
	Parameter knee_rot : (boolean)If true, noise will also be added to the knee joints.This is default false
					since adding noise to this joint can lead the character to instantly fall over.
	*/
	
	double range = radian;

	//Rotate root only about y-axis which points upwards. (x-axis points along direction of character movement). 
	tQuaternion root_rotate = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), cMathUtil::RandDouble(-range, range));
	RotateRoot(root_rotate);



	/*
	First, we do interpolation of velocities between 0 and reference motion velocity. 
	This interpolation only changes velocity, not direction of rotation. 
	*/

	//Interpolate root velocities
	tVector vel = cKinTree::GetRootVel(mVel);
	vel = interp * vel;
	cKinTree::SetRootVel(vel, mVel);

	tVector ang_vel = cKinTree::GetRootAngVel(mVel);
	ang_vel = interp * ang_vel;
	cKinTree::SetRootAngVel(ang_vel, mVel);

	//Interpolate joint velocities (j=0 is root so no need to loop through this)
	for (int j =1 ; j < GetNumJoints(); ++j)
	{
		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
	;
		tVector vel = tVector::Zero();
		vel.segment(0, param_size) = mVel.segment(param_offset, param_size);
		vel = interp * vel;
		mVel.segment(param_offset, param_size) = vel.segment(0, param_size);
	}
	
	//Add noise to pose (except root which is j = 0)
	for (int j = 1; j < GetNumJoints(); ++j)
	{
		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
		cKinTree::eJointType joint_type = GetJointType(j);

		if(joint_type == cKinTree::eJointTypeRevolute){
			/*
			When param_size = 1, these are revolute joints (1D rotations). 
			j==4 and j==10 correspond to the right knee and left knee, respectively. 
			We do not add to them by default unless knee_rot=True as this can cause
			the character to immediately fall over. Still need to do more testing. . 
			*/
			if (!(j == 4 || j == 10) || knee_rot) {
				mPose(param_offset) = mPose(param_offset) + cMathUtil::RandDouble(-range, range);
			}
		}
		else if(joint_type==cKinTree::eJointTypeSpherical) {
			/*
			For spherical joints, we sample random angle by sampling radians in (-range,range)
			for psi, theta, and phi. We do not add noise to the ankle or hip joints as this can
			cause the character to reset to a state immediately stands over (i.e. reference is standing
			but changing the hip/ankle causes it to not be standing and since velocity is initialized
			as if the character was standing, the character immediately falls over).
			*/
			if (j != 3 && j != 5 && j != 9 && j != 11) {// Don't change ankle or hip 
				double rand_psi = cMathUtil::RandDouble(-range, range);
				double rand_theta = cMathUtil::RandDouble(-range, range);
				double rand_phi = cMathUtil::RandDouble(-range, range);

				tQuaternion rand_rotate = cMathUtil::EulerToQuaternion(tVector(rand_psi, rand_theta, rand_phi, 0));


				tQuaternion rot = cMathUtil::VecToQuat(mPose.segment(param_offset, param_size));
				rot = rand_rotate * rot;
				mPose.segment(param_offset, param_size) = cMathUtil::QuatToVec(rot).segment(0, param_size);

				if (rot_vel_w_pose) {
					/*
					This if statement rotates the velocity with the same rotation as pose. 
					Not entirely sure if this actually does anything so this is disabled by default.
					*/
					tQuaternion vel = cMathUtil::VecToQuat(mVel.segment(param_offset, param_size));
					vel = rand_rotate * vel;
					mVel.segment(param_offset, param_size) = cMathUtil::QuatToVec(vel).segment(0, param_size);

				
				}
			}
		}
	}
	//Add noise to velocity with both direction and rate. 
	if (vel_noise) {
		/*
		The code below adds random noise to the linear velocity of the root. 
		This is commented out since no noise is added to the position of the root. 

		tVector vel = cKinTree::GetRootVel(mVel);
		vel(0) += cMathUtil::RandDouble(-linear_vel_noise , linear_vel_noise);
		vel(1) += cMathUtil::RandDouble(-linear_vel_noise, linear_vel_noise);
		vel(2) += cMathUtil::RandDouble(-linear_vel_noise, linear_vel_noise);
		cKinTree::SetRootVel(vel, mVel);
		*/

		//Noise added to angular velocity of root since noise added to rotation of root but can also comment out.
		tQuaternion ang_vel = cMathUtil::VecToQuat(cKinTree::GetRootAngVel(mVel));
		double rand_psi = cMathUtil::RandDouble(-range, range);
		double rand_theta = cMathUtil::RandDouble(-range, range);
		double rand_phi = cMathUtil::RandDouble(-range, range);

		tQuaternion rand_rotate = cMathUtil::EulerToQuaternion(tVector(rand_psi, rand_theta, rand_phi, 0));
		ang_vel = rand_rotate * ang_vel;  
		cKinTree::SetRootAngVel(cMathUtil::QuatToVec(ang_vel), mVel);
		

		/*
		The code for adding noise to the angular velocities of the joints follows
		the same outline as the code above for adding noise to the positions. 
		*/
		for (int j = 1; j < GetNumJoints(); ++j)
		{
			int param_offset = GetParamOffset(j);
			int param_size = GetParamSize(j);
			cKinTree::eJointType joint_type = GetJointType(j);

			if (joint_type == cKinTree::eJointTypeRevolute) {
				if (!(j == 4 || j != 10) || knee_rot) {
					
					//For revolute joints, noise not added to knee joints unless knee_rot=True
					
					mVel(param_offset) = mVel(param_offset) + cMathUtil::RandDouble(-range, range);
				}
			}
			else if (joint_type == cKinTree::eJointTypeSpherical) {
				/*
				Noise added to all spherical joints except hip and ankle. 
				*/
				if (j != 3 && j != 5 && j != 9 && j != 11) {// Don't change ankle or hip 
					double rand_psi = cMathUtil::RandDouble(-range, range);
					double rand_theta = cMathUtil::RandDouble(-range, range);
					double rand_phi = cMathUtil::RandDouble(-range, range);

					tQuaternion rand_rotate = cMathUtil::EulerToQuaternion(tVector(rand_psi, rand_theta, rand_phi, 0));

					tQuaternion vel = cMathUtil::VecToQuat(mVel.segment(param_offset, param_size));
					vel = rand_rotate*vel;//Change axis
					mVel.segment(param_offset, param_size) = cMathUtil::QuatToVec(vel).segment(0, param_size);
				}
			}
		}
	}
	cKinTree::PostProcessPose(mJointMat, mPose);
}

void cKinCharacter::SetController(const std::shared_ptr<cKinController>& ctrl)
{
	RemoveController();
	mController = ctrl;

	Pose();
	mPose0 = GetPose();
	mVel0 = GetVel();
}

const std::shared_ptr<cKinController>& cKinCharacter::GetController() const
{
	return mController;
}

void cKinCharacter::RemoveController()
{
	if (HasController())
	{
		mController.reset();
	}
}

bool cKinCharacter::HasController() const
{
	return mController != nullptr;
}

tVector cKinCharacter::GetCycleRootDelta() const
{
	tVector delta = tVector::Zero();
	const cMotionController* ctrl = dynamic_cast<cMotionController*>(mController.get());
	if (ctrl != nullptr)
	{
		delta = ctrl->GetCycleRootDelta();
	}
	return delta;
}

void cKinCharacter::CalcPose(double time, Eigen::VectorXd& out_pose) const
{
	tVector root_delta = tVector::Zero();

	if (HasController())
	{
		mController->CalcPose(time, out_pose);

		tVector root_pos = cKinTree::GetRootPos(out_pose);
		tQuaternion root_rot = cKinTree::GetRootRot(out_pose);
		root_rot = mOriginRot * root_rot;
		root_rot = cMathUtil::StandardizeQuat(root_rot);

		root_pos = cMathUtil::QuatRotVec(mOriginRot, root_pos);
		root_pos += mOrigin;

		cKinTree::SetRootPos(root_pos, out_pose);
		cKinTree::SetRootRot(root_rot, out_pose);
	}
	else
	{
		out_pose = mPose0;
	}
}

void cKinCharacter::CalcIndexPose(int index, Eigen::VectorXd& out_pose) const
{
	tVector root_delta = tVector::Zero();

	if (HasController())
	{
		mController->CalcIndexPose(index, out_pose);
		tVector root_pos = cKinTree::GetRootPos(out_pose);
		tQuaternion root_rot = cKinTree::GetRootRot(out_pose);
		root_rot = mOriginRot * root_rot;
		root_rot = cMathUtil::StandardizeQuat(root_rot);

		root_pos = cMathUtil::QuatRotVec(mOriginRot, root_pos);
		root_pos += mOrigin;
		cKinTree::SetRootPos(root_pos, out_pose);
		cKinTree::SetRootRot(root_rot, out_pose);
	}
	else
	{
		out_pose = mPose0;
	}
}


void cKinCharacter::CalcVel(double time, Eigen::VectorXd& out_vel) const
{
	if (HasController())
	{
		mController->CalcVel(time, out_vel);

		tVector root_vel = cKinTree::GetRootVel(out_vel);
		tVector root_ang_vel = cKinTree::GetRootAngVel(out_vel);
		root_vel = cMathUtil::QuatRotVec(mOriginRot, root_vel);
		root_ang_vel = cMathUtil::QuatRotVec(mOriginRot, root_ang_vel);

		cKinTree::SetRootVel(root_vel, out_vel);
		cKinTree::SetRootAngVel(root_ang_vel, out_vel);
	}
	else
	{
		out_vel = Eigen::VectorXd::Zero(GetNumDof());
	}
}

void cKinCharacter::CalcIndexVel(int index, Eigen::VectorXd& out_vel) const
{
	if (HasController())
	{
		mController->CalcIndexVel(index, out_vel);

		tVector root_vel = cKinTree::GetRootVel(out_vel);
		tVector root_ang_vel = cKinTree::GetRootAngVel(out_vel);
		root_vel = cMathUtil::QuatRotVec(mOriginRot, root_vel);
		root_ang_vel = cMathUtil::QuatRotVec(mOriginRot, root_ang_vel);

		cKinTree::SetRootVel(root_vel, out_vel);
		cKinTree::SetRootAngVel(root_ang_vel, out_vel);
	}
	else
	{
		out_vel = Eigen::VectorXd::Zero(GetNumDof());
	}
}




void cKinCharacter::CalcAcc(double time, Eigen::VectorXd& out_acc) const
{
	Eigen::VectorXd vel0;
	Eigen::VectorXd vel1;
	CalcVel(time - gDiffTimeStep, vel0);
	CalcVel(time, vel1);
	out_acc = (vel1 - vel0) / gDiffTimeStep;
}

bool cKinCharacter::IsMotionOver() const
{
	bool over = true;
	if (HasController())
	{
		over = mController->IsMotionOver();
	}
	return over;
}

void cKinCharacter::BlendFrames(const cMotion::tFrame* a, const cMotion::tFrame* b, double lerp, cMotion::tFrame* out_frame) const
{
	cKinTree::LerpPoses(mJointMat, *a, *b, lerp, *out_frame);
}

void cKinCharacter::BlendVel(const Eigen::VectorXd* a, const Eigen::VectorXd* b, double lerp, Eigen::VectorXd* out_vel) const
{
	cKinTree::LerpVels(*a, *b, lerp, *out_vel);
}

void cKinCharacter::CalcFrameVel(const cMotion::tFrame* a, const cMotion::tFrame* b, double timestep, cMotion::tFrame* out_vel) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::CalcVel(joint_mat, *a, *b, timestep, *out_vel);
}

void cKinCharacter::PostProcessPose(cMotion::tFrame* out_frame) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::PostProcessPose(joint_mat, *out_frame);
}