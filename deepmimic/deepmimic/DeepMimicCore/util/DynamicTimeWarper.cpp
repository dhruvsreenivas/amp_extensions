#include "DynamicTimeWarper.h"
#include <iostream>
double cDynamicTimeWarper::DefaultCost(const Eigen::VectorXd* data0, const Eigen::VectorXd* data1)
{
	assert(data0->size() == data1->size());
	double cost = (*data0 - *data1).squaredNorm();
	return cost;
}

cDynamicTimeWarper::cDynamicTimeWarper()
{
	mCostFunc = cDynamicTimeWarper::DefaultCost;

	mSampleCount0 = 0;
	mSampleCount1 = 0;
	mComputedIndexBuffer = false;
	mData0 = Eigen::MatrixXd(0, 0);
	mData1 = Eigen::MatrixXd(0, 0); 
	mCostBuffer = Eigen::MatrixXd(0, 0);
	mIndexBuffer = Eigen::VectorXi();
	mBacktrackPath = Eigen::VectorXd();
}

cDynamicTimeWarper::~cDynamicTimeWarper()
{
}

void cDynamicTimeWarper::Init(int dim0, int dim1, int buffer_size, tCostFunc cost_func)
{
	if (cost_func != nullptr)
	{
		mCostFunc = cost_func;
	}
	else
	{
		mCostFunc = cDynamicTimeWarper::DefaultCost;
	}

	mSampleCount0 = 0;
	mSampleCount1 = 0;
	mComputedIndexBuffer = false;
	mData0 = Eigen::MatrixXd(buffer_size, dim0);
	mData1 = Eigen::MatrixXd(buffer_size, dim1);
	mCostBuffer = Eigen::MatrixXd(buffer_size, buffer_size);
	mIndexBuffer = Eigen::VectorXi(buffer_size);
	mBacktrackPath = Eigen::VectorXd(buffer_size);
}

void cDynamicTimeWarper::Reset()
{
	mSampleCount0 = 0;
	mSampleCount1 = 0;
	mComputedIndexBuffer = false;
}

int cDynamicTimeWarper::GetBufferSize()
{
	return static_cast<int>(mIndexBuffer.size());
}

int cDynamicTimeWarper::GetDim0() const
{
	return static_cast<int>(mData0.cols());
}

int cDynamicTimeWarper::GetDim1() const
{
	return static_cast<int>(mData1.cols());
}

int cDynamicTimeWarper::GetSampleCount0() const
{
	return mSampleCount0;
}

int cDynamicTimeWarper::GetSampleCount1() const
{
	return mSampleCount1;
}

void cDynamicTimeWarper::AddSample0(const Eigen::VectorXd& data)
{
	assert(data.size() == GetDim0());
	mData0.row(mSampleCount0) = data;
	mSampleCount0 += 1;

	int buffer_size = GetBufferSize();
	if (mSampleCount0 > buffer_size)
	{
		printf("Time warper buffer 0 overflow, capacity: %d\n", buffer_size);
		throw;
	}
}

void cDynamicTimeWarper::AddSample1(const Eigen::VectorXd& data)
{
	assert(data.size() == GetDim1());
	mData1.row(mSampleCount1) = data;
	mSampleCount1 += 1;

	int buffer_size = GetBufferSize();
	if (mSampleCount1 > buffer_size)
	{
		printf("Time warper buffer 1 overflow, capacity: %d\n", buffer_size);
		throw;
	}
}

Eigen::VectorXd cDynamicTimeWarper::GetSample0(int i) const
{
	return mData0.row(i);
}

Eigen::VectorXd cDynamicTimeWarper::GetSample1(int i) const
{
	return mData1.row(i);
}

double cDynamicTimeWarper::CalcAlignment()
{
	int n = GetSampleCount0();
	int m = GetSampleCount1();
	mCostBuffer.fill(std::numeric_limits<double>::infinity());
	mCostBuffer(0, 0) = 0.0;
	
	for (int i = 1; i < n; ++i)
	{
		Eigen::VectorXd data0 = GetSample0(i);
		for (int j = 1; j < m; ++j)
		{
			Eigen::VectorXd data1 = GetSample1(j);

			double val = CalcCost(data0, data1);
			double cost0 = mCostBuffer(i - 1, j - 1);
			double cost1 = mCostBuffer(i - 1, j);
			double cost2 = mCostBuffer(i, j - 1);

			double min_cost = std::min(cost0, std::min(cost1, cost2));
			double curr_cost = val + min_cost;

			mCostBuffer(i, j) = curr_cost;
		}
	}
	CalcIndexBuffer();//new added here.
	return GetMinAlignmentCost();
}

void cDynamicTimeWarper::CalcIndexBuffer() {
	if (mComputedIndexBuffer) {
		return;
	}
	int n = GetSampleCount0();
	int m = GetSampleCount1();
	int i = n - 1;
	int j = m - 1;

	double i_running_cost = 0;
	while (i >= 1 && j >= 1)
	{
		mIndexBuffer(i) = j;

		double current_total_cost = mCostBuffer(i, j);
		double cost0 = mCostBuffer(i - 1, j - 1);
		double cost1 = mCostBuffer(i - 1, j);
		double cost2 = mCostBuffer(i, j - 1);

		if (cost0 <= cost1 && cost0 <= cost2)
		{
			i_running_cost += current_total_cost - cost0;
			mBacktrackPath(i) = i_running_cost;
			i_running_cost = 0;

			i = i - 1;
			j = j - 1;
		}
		else if (cost1 <= cost2)
		{
			i_running_cost += current_total_cost - cost1;
			mBacktrackPath(i) = i_running_cost;
			i_running_cost = 0;
			i = i - 1;

		}
		else
		{
			i_running_cost += current_total_cost - cost2;
			j = j - 1;
		}
		i = std::max(0, i);
		j = std::max(0, j);
	}
	mIndexBuffer.segment(0, i + 1).fill(0);

	Eigen::VectorXd data1 = GetSample1(j);
	for (int k = 0; k < i + 1;  k++) {
		Eigen::VectorXd data1 = GetSample0(k);
		mBacktrackPath(k) = CalcCost(data1, data1);
	}
	mComputedIndexBuffer = true;
}

void cDynamicTimeWarper::AddRemainderCost(double cost) 
{	
	if (!mComputedIndexBuffer) {
		CalcIndexBuffer();
	}
	int n = GetSampleCount0();
	/*
	Important to segment here since for resetting, we never actually clear the vectors. Everything
	is based on the sample count. 
	*/
	Eigen::VectorXd costs = mBacktrackPath.segment(0, n);
	mBacktrackPath.segment(0, n) += (costs / costs.sum()) * cost;
}
const Eigen::VectorXi& cDynamicTimeWarper::CalcMinCostPath()
{
	if (!mComputedIndexBuffer) {
		CalcIndexBuffer();
	}
	return GetMinCostPath();
}

const Eigen::VectorXd& cDynamicTimeWarper::CalcDTWBacktrackPath()
{
	if (!mComputedIndexBuffer) {
		CalcIndexBuffer();
	}	
	return GetDTWBacktrackPath();
}
const Eigen::VectorXd& cDynamicTimeWarper::GetDTWBacktrackPath() const 
{

	int n = GetSampleCount0();
	return mBacktrackPath.segment(0, n);
}

const Eigen::VectorXi& cDynamicTimeWarper::GetMinCostPath() const
{
	int n = GetSampleCount0();
	return mIndexBuffer.segment(0, n);
}

double cDynamicTimeWarper::GetMinAlignmentCost() const
{
	int n = GetSampleCount0();
	int m = GetSampleCount1();
	double min_cost = mCostBuffer(n - 1, m - 1);
	return min_cost;
}

double cDynamicTimeWarper::CalcCost(const Eigen::VectorXd& data0, const Eigen::VectorXd& data1) const
{
	double cost = mCostFunc(&data0, &data1);
	return cost;
}