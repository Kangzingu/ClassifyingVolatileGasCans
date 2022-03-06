#pragma once
#ifndef _ELEMENT_H
#define _ELEMENT_H
class Element
{
public:
	Element();
	~Element();
	Element(int st_x, int st_y, int w,int h, int type, int value);

	//private:

	int start_x;
	int start_y;

	int end_x;
	int end_y;

	int type;//0-4
	float value;

};
#endif
