#include <cstdio>
#include <iostream>
#include <cstring>

float a[267][267][15] = {0};

int main(){
	for (int i = 0; i < 60; ++i){
        std::string t = "trainingdata/tf" + std::to_string(i) + ".txt";
        FILE *f = fopen(t.c_str(), "r");
        memset(a, 0, sizeof(a));
        float x, y, z[20];
        int imx, imy;
        while(~fscanf(f, "%f%f%f%f%f%f%f%f%f%f%f", &x, &y, &z[0], &z[1], &z[2], &z[3], &z[4], &z[5], &z[6], &z[7], &z[8])){
        	imx = int(x - 0.5f);
        	imy = int(y - 0.5f);
        	for (int j = 0; j < 9; j++)a[imx][imy][j] = z[j];
        }
    	fclose(f);
    	std::string ta = "Data/tf" + std::to_string(i) + "_in.txt";
    	f = fopen(ta.c_str(), "w");
    	fprintf(f,"[\n");
    	for (int j = 0; j < 256; j++){
    		fprintf(f,"[");
    		for (int k = 0; k < 256; k++){
    			fprintf(f,"[");
    			for (int o = 0; o < 9; o++){
    				if (o == 1) continue;
    				fprintf(f,"%f", a[j][k][o]);
    				if (o < 8)fprintf(f,",");
    			}
    			fprintf(f,"]");
    			if (k < 255)fprintf(f,",\n");
    		}
    		fprintf(f,"]");
    		if(j < 255)fprintf(f,",\n");
    	}
    	fprintf(f,"]\n");
    	fclose(f);
    	std::string tb = "Data/tf" + std::to_string(i) + "_out.txt";
    	f = fopen(tb.c_str(), "w");
    	fprintf(f,"[\n");
    	for (int j = 0; j < 256; j++){
    		fprintf(f,"[");
    		for (int k = 0; k < 256; k++){
    			fprintf(f,"[%f]", a[j][k][1]);
    			if (k < 255)fprintf(f,",");
    		}
    		fprintf(f,"]");
    		if(j < 255)fprintf(f,",\n");
    	}
    	fprintf(f,"]\n");
    	fclose(f);
        
    }
	return 0;
}