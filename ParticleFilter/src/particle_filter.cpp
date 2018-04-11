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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set number of particles
	num_particles_ = 1000; //1000;

	// Set normal distributions for states
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Set random generator
	default_random_engine gen;

	// Loop over number of particles and initialize each of them with noise
	for (int i = 0; i < num_particles_; ++i) {
		
		// Create new particle
		Particle p;

		// Set data
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;

		particles_.push_back(p);
		weights_.push_back(1);
	}

	//printParticles();

	// Set boolean to be initialized
	is_initialized_ = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate){
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Predict
	std::cout << "Predict with t " << delta_t << " v " << velocity << " yr " << yaw_rate << std::endl;

	// Set normal distributions for odometry measurements
	normal_distribution<double> dist_v(velocity, std_pos[0]);
	normal_distribution<double> dist_yr(yaw_rate, std_pos[1]);

	// Set random generator
	default_random_engine gen;

	// Loop over all particles and shift them according to the input information
	for(int i = 0; i < num_particles_; i++){

		// Grab particle
		Particle & p = particles_[i];

		// Sample odometry measurement
		float z_v = dist_v(gen);
		float z_yr = dist_yr(gen);

		// Update orientation of particle
		p.theta += z_yr * delta_t;

		//avoid division by zero
		if(fabs(z_yr) > 0.001) {
			p.x += z_v / z_yr * ( sin (p.theta + z_yr * delta_t) - sin (p.theta));
			p.y += z_v / z_yr * ( cos (p.theta) - cos (p.theta + z_yr * delta_t));
		}
		else{
			p.x += z_v * delta_t * cos(p.theta);
			p.y += z_v * delta_t * sin(p.theta);
		}

	}

	//printParticles();

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks){
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	// Buffer for total sum of weights
	double sum_of_weights = 0.0;

	// Transform
	for(int i = 0; i < num_particles_; ++i){

		// Grab particle
		Particle & p = particles_[i];

		// Create transformed landmark vector
		std::vector<LandmarkObs> observations_map_frame;

		// Calculate x and y term for particle
		float sin_term = sin(p.theta);
		float cos_term = cos(p.theta);

		// Loop over observations and tranform them in map frame
		for(int j = 0; j < observations.size(); ++j){

			// Create new LandmarksObs
			LandmarkObs obs_m;

			// Fill information with map coordinates
			obs_m.x = p.x + cos_term * observations[j].x - sin_term * observations[j].y;
			obs_m.y = p.y + sin_term * observations[j].x + cos_term * observations[j].y;

			//std::cout << "Observation " << j << " " << observations[j].x << " " << observations[j].y << std::endl;
			//std::cout << "Transformed " << j << " " << obs_m.x << " " << obs_m.y << std::endl;

			// Push back transformed observation
			observations_map_frame.push_back(obs_m);
		}
		// Data association

		// Loop over all predicted landmarks
		double weight = 1.0;
		for(int k = 0; k < observations_map_frame.size(); k++){

			// Grab predicted landmark
			LandmarkObs & pre = observations_map_frame[k];

			double min_dist = sensor_range;
			int min_index = -1;

			// Loop over all observations
			for(int j = 0; j < map_landmarks.landmark_list.size(); j++){

				double distance = dist(pre.x, pre.y,
					map_landmarks.landmark_list[j].x_f,
					map_landmarks.landmark_list[j].y_f);

				if(distance < min_dist){

					min_dist = distance;
					min_index = j;
				}
			}

			// Get weight
			double x_off = abs(pre.x - map_landmarks.landmark_list[min_index].x_f);
			double y_off = abs(pre.y - map_landmarks.landmark_list[min_index].y_f);
			weight *= getWeight(x_off, y_off, std_landmark[0], std_landmark[1]);

			//std::cout << "w " << weight << " (x,y) " << x_off << " " << y_off << std::endl;

		}

		p.weight = weight;

		//std::cout << "W " << weight << std::endl;

		sum_of_weights += weight;
	}

	// Normalize weights
	cum_weights_.clear();
	cum_weights_.push_back(0.0);
	for(int i = 0; i < num_particles_; ++i){

		particles_[i].weight /= sum_of_weights;

		//std::cout << "W " << i << " " << particles_[i].weight << std::endl;

		cum_weights_.push_back(cum_weights_.back() + particles_[i].weight);

	}

}

void ParticleFilter::resample(){
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	uniform_real_distribution<double> distribution(0.0,1.0);

	std::vector<Particle> resampled_particles;
	for(int i = 0; i < num_particles_ ; i++){

		double val = distribution(gen);

		int index = num_particles_;
		for(int j = 0; j < num_particles_; j++){

			if(val > cum_weights_[j] && val <= cum_weights_[j+1]){
				index = j;
				break;
			}
		}

		Particle p = particles_[index];
		p.id = i;
		resampled_particles.push_back(p);

		//std::cout << i << " " << index << std::endl;
	}

	particles_ = resampled_particles;

	//printParticles();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y){
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best){
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best){
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best){
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::printParticles(){

	for(int i = 0; i < num_particles_; i++){

		Particle & p = particles_[i];
		cout << "P " << p.id << " " << p.x << " " << p.y << " " << p.theta
			<< " " << p.weight << endl;
	}
}

double ParticleFilter::getWeight(const float x_off, const float y_off, const float std_x, const float std_y){

	return 1 / ( 2 * M_PI * std_x * std_y) * std::exp(- (x_off*x_off/std_x/std_x + y_off*y_off/std_y/std_y) / 2);
}
