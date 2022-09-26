#pragma once

#include <random>
#include "Eigen/StdVector"
class cRand
{
public:
	cRand();
	cRand(unsigned long int seed);
	virtual ~cRand();

	virtual double RandDouble();
	virtual double RandDouble(double min, double max);
	virtual Eigen::VectorXd RandDoubleEigen(int size, double min, double max); //new
	virtual double RandDoubleExp(double lambda);
	virtual double RandDoubleNorm(double mean, double stdev);
	virtual int RandInt();
	virtual int RandInt(int min, int max);
	virtual int RandUint();
	virtual int RandUint(unsigned int min, unsigned int max);
	virtual int RandIntExclude(int min, int max, int exc);
	virtual void Seed(unsigned long int seed);
	virtual int RandSign();
	virtual bool FlipCoin(double p = 0.5);

private:
	std::default_random_engine mRandGen;
	std::mt19937 mGen; //new
	std::uniform_real_distribution<double> mRandDoubleDist;
	std::normal_distribution<double> mRandDoubleDistNorm;
	std::uniform_int_distribution<int> mRandIntDist;
	std::uniform_int_distribution<unsigned int> mRandUintDist;
};