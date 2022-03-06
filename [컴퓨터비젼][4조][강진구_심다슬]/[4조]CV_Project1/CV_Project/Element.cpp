#include "Element.h"

Element::Element()
{

}

Element::~Element()
{
}
Element::Element(int st_x, int st_y, int ed_x, int ed_y, int _type, int _value)
{

	start_x = st_x;
	start_y = st_y;
	this->end_x = ed_x;
	this->end_y = ed_y;

	//switch (_type)
	//{
	//case 0:
	//	end_x = start_x + 2*w;
	//	end_y = start_y + 1*h;
	//	break;
	//case 1:
	//	end_x = start_x + 3*w;
	//	end_y = start_y + 1*h;
	//	break;
	//case 2:
	//	end_x = start_x + 1*w;
	//	end_y = start_y + 2*h;
	//	break;
	//case 3:
	//	end_x = start_x + 1*w;
	//	end_y = start_y + 3*h;
	//	break;
	//case 4:
	//	end_x = start_x + 2*w;
	//	end_y = start_y + 2*h;
	//	break;

	//default:
	//	break;
	//}


	value = _value;
	type = _type;

}
