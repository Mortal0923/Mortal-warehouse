#pragma once

#include "Ball.hpp"

class Basket :
		public Ball
{
public:
	std::vector<Ball> containedBalls_;

	explicit Basket(Ball &ball);
};