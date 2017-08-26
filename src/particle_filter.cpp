/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::ostream& operator<<(std::ostream& os, const Particle& a ) {
	os << a.x << ", " << a.y << ", " << a.theta;
	return os;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100; // ? 
	weights.resize(num_particles, 1);
	std::default_random_engine gen;
	std::normal_distribution<double> x_dist(x, std[0]);
	std::normal_distribution<double> y_dist(y, std[1]);
	std::normal_distribution<double> t_dist(theta, std[2]);
	for(int i=0; i < num_particles; i++) {
		Particle temp = Particle();
		temp.id = i;
		temp.x = x_dist(gen);
		temp.y = y_dist(gen);
		temp.theta = t_dist(gen);
		particles.push_back(temp);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// implement motion model which is bicycle motion model
	
	// noise modeling 0 mean stddev same as gps signal stddev as that is what is passed
	std::default_random_engine gen;
	std::normal_distribution<double> x_dist(0, std_pos[0]);
	std::normal_distribution<double> y_dist(0, std_pos[1]);
	std::normal_distribution<double> t_dist(0, std_pos[2]);

	// update particles with motion model + noise
	for (Particle& p : particles) {
		if(fabs(yaw_rate) > 0.0001) {
			p.x = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta)) + x_dist(gen);
			p.y = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t)) + y_dist(gen);
			p.theta = p.theta + yaw_rate*delta_t + t_dist(gen);
		} else {
			p.x = p.x + velocity*delta_t*cos(p.theta) + x_dist(gen);
			p.y = p.y + velocity*delta_t*sin(p.theta) + y_dist(gen);
			p.theta = p.theta + t_dist(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predict measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	auto most_likelyid = [](const LandmarkObs& a, const std::vector<LandmarkObs>& list)->int {
		double ldist = -1;
		int closest = -1;
		for(int i=0; i<list.size(); i++) {
			const LandmarkObs& l = list[i];
			double distance = dist(a.x, a.y, l.x, l.y);			
			if(ldist < 0) {
				ldist = distance;
				closest = i;
				continue;
			}

			if(distance < ldist) {
				ldist = distance;
				closest = i;
			}
		}
		return closest;
	};

	for(int i=0; i<observations.size(); i++) {
		observations[i].id = most_likelyid(observations[i], predicted);
	}
}

void vehicleToMap(const Particle& p, const std::vector<LandmarkObs>& obs, std::vector<LandmarkObs>* const mapObs) {
	std::vector<LandmarkObs> ret;
	for(int i=0; i< obs.size(); i++) {
		// xconv = particle_pos[0] + vcoord[0]*math.cos(particle_theta) - vcoord[1]*math.sin(particle_theta)
		// yconv = particle_pos[1] + vcoord[0]*math.sin(particle_theta) + vcoord[1]*math.cos(particle_theta)
		(*mapObs)[i].x = p.x + obs[i].x*cos(p.theta) - obs[i].y*sin(p.theta);
		(*mapObs)[i].y = p.y + obs[i].x*sin(p.theta) + obs[i].y*cos(p.theta);
	}
}

long double multivariate_guassian(const LandmarkObs& predicted, const LandmarkObs& actual, const double std_landmark[]) {
	// def likelihood(obs, landmark):
    // const = 1/(2*math.pi*std_x*std_y)
    // exponent = math.pow((obs[0] - landmark[1][0]), 2)/(2*std_x*std_x) + math.pow((obs[1] - landmark[1][1]), 2)/(2*std_y*std_y)
	// return const*math.exp(-exponent)
	const static long double multiplier = 1.0/(2*M_PI*std_landmark[0]*std_landmark[1]);
	long double exponent = (pow((double(predicted.x) - actual.x), 2.0)/(2*std_landmark[0]*std_landmark[0])) + 
						(pow((double(predicted.y) - actual.y), 2.0)/(2*std_landmark[1]*std_landmark[1]));
	return multiplier*exp(-exponent);
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// transform observations to map coordinates for each particle

	// using observations and map_landmarks update weights of each of the particles which suggests which are likely
	// all particles have been updated with motion model in prediction step hence now this is about finding likelihood
	// multiplying likelihood and prediction taking into consideration multi-variate gaussian distribution as likelihood
	// and motion model both are gaussian distributions. Then normalizing the weights which gives the posterior weights.

	std::vector<LandmarkObs> mapObs;
	mapObs.resize(observations.size());
	double sum_probability = 0.0;
	for(int i=0; i < particles.size(); i++) {
		const Particle& p = particles[i];
		// retrieved observations map
		vehicleToMap(p, observations, &mapObs);	

		// associate most likely landmark by the logic of finding least distance
		std::vector<LandmarkObs> landmarks;
		for(auto x : map_landmarks.landmark_list) {
			if (dist(x.x_f, x.y_f, p.x, p.y) < sensor_range) {
				LandmarkObs l = LandmarkObs();
				l.id = x.id_i;
				l.x = x.x_f;
				l.y = x.y_f;
				landmarks.push_back(l);
			}
		}
		dataAssociation(landmarks, mapObs);

		// find probability of observation by multi variate guassian distribution around landmark
		long double particle_prob = 1.0;
		for(LandmarkObs& o: mapObs) {
			particle_prob *= multivariate_guassian(o, landmarks[o.id], std_landmark);
		}

		sum_probability += particle_prob;
		weights[i] = particle_prob;
	}
	// normalize to update weights
	for(int i=0; i < weights.size(); i++) {
		weights[i] = weights[i]/sum_probability;
		particles[i].weight = weights[i];
	};
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device rd;
 	mt19937 gen(rd());
	discrete_distribution<> d(weights.begin(), weights.end());
	
	vector<Particle> sampled;
	sampled.resize(particles.size());
	for(int i=0; i<num_particles; i++) {
		sampled[i] = particles[d(gen)];
	}
	particles = sampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
