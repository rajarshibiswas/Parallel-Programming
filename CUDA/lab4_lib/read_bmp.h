/*
---------------------------------------------------------------------
---------------------------------------------------------------------
FILE: read_bmp.h
	Contains all the declarations of functionalities supported 
	by BMP Reader library. Include the header file into the project
	to link the functions into your project space
Library: read_bmp.o
---------------------------------------------------------------------
---------------------------------------------------------------------
*/
#ifndef _BMP_READ_LIB_
#define _BMP_READ_LIB_

typedef unsigned int uint32_t;
typedef signed int   int32_t;
typedef unsigned short uint16_t;
typedef signed short int16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;

class bmp_image{
	private:
		uint8_t *hdr_buffer;
	public:
		uint32_t image_width;
		uint32_t image_height;
		uint32_t num_pixel;
//----------------------------------------------------				
//FUNC: read_bmp_file(FILE *bmp_image_file)
//DESCP:
//	Reads the image file from FILE provided, 
//	Extracts the header information of BMP file
//	Allocates memory with pixel values and returns
//	the data
//NOTE:	Will work only on .bmp image files
//----------------------------------------------------
		void* read_bmp_file(FILE *bmp_file);


//----------------------------------------------------				
//FUNC: write_bmp_file(FILE *out_file, uint8_t *bmp_data)
//DESCP:
//	Writes the image file to FILE * provided, 
//	Adds the header information of BMP file
//	Copies pixel values from buffer provided
//NOTE:	Will create .bmp type file formats
//----------------------------------------------------
		void write_bmp_file(FILE *out_file, uint8_t *bmp_data);

		bmp_image();
};


#endif
