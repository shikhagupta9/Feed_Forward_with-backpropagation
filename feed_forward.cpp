#include <sstream>
#include <fstream>
#include <iostream>
#include<vector>
#include <cmath>
#include<float.h>
#define Attributes 6
#define classification 3
#define Alpha 0.05
#define Gamma 0.01
using namespace std;


void print_out(const vector<double> &v);

//to conver output into vector form
void classify(const vector<double>& data_output,vector<vector<double>> &target);

//chosen activation functions are sigmoid, RELU, tanh
double sigmoid(double x);
double dSigmoid(double x);

double tanh(double x);
double dtanh(double x);

double ReLU(double x);
double dReLU(double x);

double convert_to_num(string str);

double fRand(double fMin, double fMax);

//for feed forward
double Activation(int type,double value);

//for backpropogation
double Activation_Derivative(int type,double value);
void file_read(vector<vector<double> >&dataset);

//to split data into training and testing
void split_data( vector<vector<double>> &dataset,vector<vector<double>> &validation,vector<vector<double>> &target,vector<vector<double>> &validation_target);


int main()
{
    int Instances=310,i,j;
    vector<vector<double> > dataset;
    vector <double> dataset_output;
    vector<vector<double>> validation;
    vector<vector<double>>target(Instances, vector<double>(classification,0));
    vector<vector<double>>val_target;

    file_read(dataset);

    //shuffling the dataset
    srand(unsigned(time(0)));
    random_shuffle(dataset.begin(), dataset.end());

    //creating output
    for(int i=0;i<dataset.size();i++)
    {
        dataset_output.push_back(dataset[i].back());
        dataset[i].pop_back();
    }
    classify(dataset_output,target);

    split_data(dataset, validation,target,val_target);

    Instances=dataset.size();

    int option_chosen=0;
    cout<<"CHOOSE THE TYPE OF NETWORK"<<endl;
    cout<<"Enter 0 -> FULLY CONNECTED network \nEnter 1-> CUSTOM network"<<endl;
    cin>>option_chosen;
    cout<<endl;

    int epoch,num_of_HiddenLayers;
    cout<<"Enter the no. of epochs"<<endl;
    cin>>epoch;
    cout<< "Enter number of HIDDEN LAYERS"<<endl;
    cin>>num_of_HiddenLayers;
    cout<<endl;

    cout<<"Activation Functions are : "<<endl;
    cout<<"1. Sigmoid "<<endl;
    cout<<"2. ReLU"<<endl;
    cout<<"3. Tanh"<<endl;
    cout<<endl;

    cout<<"Enter No. of Neurons and Activation Functions for each layer"<<endl;
    cout<<endl;

    //default setting of these
    int Layer_info[num_of_HiddenLayers+2][2];  //total layers will be hidden layers, input layer and output layer
    for(i=1;i<=num_of_HiddenLayers;i++)
    {
        cout<< i << "th hidden layer - " <<endl;
        cout<<"No. of neurons  ";
        cin>>Layer_info[i][0];
        cout<<endl;
        cout<<"Enter 1 for Sigmoid, 2 for ReLU and 3 for Tanh  ";
        cin>>Layer_info[i][1];
        cout<<endl;
    }
    int Output_Activation;
    cout<<"Enter Activation for output layer"<<endl;
    cin>>Output_Activation;


    Layer_info[num_of_HiddenLayers+1][0]=classification;
    Layer_info[num_of_HiddenLayers+1][1]=Output_Activation;

    Layer_info[0][0]=Attributes;
    Layer_info[0][1]=1;

    vector<vector<vector<bool>>>user_defined_adjecency;
    for(i=0;i<num_of_HiddenLayers+1;i++)
    {
        vector<vector<bool>>v(Layer_info[i][0], vector<bool>(Layer_info[i+1][0],true));
        user_defined_adjecency.push_back(v);
    }

    int x;
    if (option_chosen==1)
       {
           cout<<"enter connectivity between all layers where layer 0 is input and last is output "<<endl;
           cout<<"if edge exists between two nodes give value one else 0"<<endl;
           for (int i=0; i<num_of_HiddenLayers+1; i++)
           {
               cout<<"adjacency matrix between layer "<<i<<" and layer "<<i+1<<endl;

               for (int j=0; j<Layer_info[i][0]; j++)
               {
                   for (int k=0; k<Layer_info[i+1][0]; k++)
                   {
                       cin>>x;
                       if(!x)
                         user_defined_adjecency[i][j][k]=false;
                   }
               }
           }
       }


    vector<string>classes; //last me
    vector<vector<double>>Layers;
    vector<vector<vector<double> > >weight;
    vector<vector<vector<double> > >delta_w;
    vector<vector<vector<double> > >best_w;
    vector<vector<double>>bias_w;
    vector<vector<double>>best_bias_w;
    

    int best_epoch=0 ;
    double min_val_error= DBL_MAX ,training_error_at_minvalerror = 0.0;

    //initialization
    for(i=0;i<num_of_HiddenLayers+2;i++)
    {
       vector<double>v(Layer_info[i][0]);
       Layers.push_back(v);
    }


    //bias initialisation
    for(i=1;i<num_of_HiddenLayers+2;i++)
    {
       vector<double>v(Layer_info[i][0]);
       bias_w.push_back(v);
    }

    for(i=0;i<bias_w.size();i++)
    {
        for(j=0;j<bias_w[i].size();j++)
        {
            bias_w[i][j] = fRand(-0.05,0.05);
        }
    }


    //weight initialisation
    for(i=0;i<num_of_HiddenLayers+1;i++)
    {
        vector<vector<double>>v(Layer_info[i][0], vector<double>(Layer_info[i+1][0]));
        weight.push_back(v);
    }

    for(int k=0;k<weight.size();k++)
    {
        for(i=0;i<weight[k].size();i++)
        {
            for(j=0;j<weight[k][i].size();j++)
            {
                if (user_defined_adjecency[k][i][j])
                {
                    weight[k][i][j] = fRand(-0.05,0.05);
                }else
                {
                     weight[k][i][j] = 0;
                }
            }
        }
    }

    vector<vector<double>> delta;
    for(i=1;i<num_of_HiddenLayers+2;i++)
    {
       vector<double>v(Layer_info[i][0],0);
       delta.push_back(v);
    }

    delta_w = weight;
    best_w=weight;
    best_bias_w=bias_w;
   



    for(int e=0;e<epoch;e++)
    {
          double error_epoch=0,error_val=0;
          for(int d=0;d<Instances;d++)
          {
                double error=0;
                Layers[0] = dataset[d];

                //calculation of hx and activation
                for(int k=1;k<num_of_HiddenLayers+2;k++)
                {
                   for(i=0;i<Layer_info[k][0];i++)
                   {
                       Layers[k][i]=0;
                       for(j=0;j<Layer_info[k-1][0];j++)
                       {
                           Layers[k][i] += Layers[k-1][j]*weight[k-1][j][i];
                       }
                       Layers[k][i] += bias_w[k-1][i];
                       Layers[k][i] = Activation(Layer_info[k][1],Layers[k][i]);
                   }
                }

                //-----------------BACK PROPOGATION----------------------------------------------------------------------

                //delta = (target-actual)f'(zin)
                for(i=0;i<classification;i++)
                {
                    delta[num_of_HiddenLayers][i] = (target[d][i]-Layers[num_of_HiddenLayers+1][i])*Activation_Derivative(Output_Activation,Layers[num_of_HiddenLayers+1][i]);

                    error =error+(target[d][i]-Layers[num_of_HiddenLayers+1][i])*(target[d][i]-Layers[num_of_HiddenLayers+1][i]);
                }

                //calculation of delta for all layers except input layer
                for(int k=num_of_HiddenLayers-1; k>=0 ; k--)
                {
                  fill(delta[k].begin(),delta[k].end(),0);
                    for(i=0;i<Layer_info[k+1][0];i++)
                    {
                         for(j=0;j<delta[k+1].size();j++)
                         {
                             delta[k][i] += delta[k+1][j]*weight[k+1][i][j];
                         }
                         delta[k][i] *= Activation_Derivative(Layer_info[k+1][1],Layers[k+1][i]);
                    }
                }

                //delta w calculation
                for(int k=num_of_HiddenLayers; k>=0 ; k--)
                {
                    for(i=0;i<Layer_info[k+1][0];i++)
                    {
                        for(j=0;j<Layer_info[k][0];j++)
                        {
                            delta_w[k][j][i]  =  Alpha*delta[k][i]*Layers[k][j];
                        }
                        //bias updation
                        bias_w[k][i] = bias_w[k][i] + Alpha*delta[k][i];
                    }
                }

                //weight updation
                for(int k=0;k<weight.size();k++)
                {
                    for(i=0;i<weight[k].size();i++)
                    {
                          for(j=0;j<weight[k][i].size();j++)
                          {
                              if (user_defined_adjecency[k][i][j])
                              {
                                 weight[k][i][j]*=(1 - 2*Gamma*Alpha);
                                  weight[k][i][j] += delta_w[k][i][j];
                                  
                              }
                          }
                    }
                }

              error=error/2;
              error_epoch=error_epoch+error;
          }


          for(int v=0;v<validation.size();v++)
          {
              double verror=0;
              Layers[0] = validation[v];

              //finding output for validation data
              for(int k=1;k<num_of_HiddenLayers+2;k++)
              {
                 for(i=0;i<Layer_info[k][0];i++)
                 {
                     Layers[k][i]=0;
                     for(j=0;j<Layer_info[k-1][0];j++)
                     {
                         Layers[k][i] += Layers[k-1][j]*weight[k-1][j][i];
                     }
                     Layers[k][i] += bias_w[k-1][i];
                     Layers[k][i] = Activation(Layer_info[k][1],Layers[k][i]);
                 }
              }

              for(i=0;i<classification;i++)
              {
                  verror =verror+(val_target[v][i]-Layers[num_of_HiddenLayers+1][i])*(val_target[v][i]-Layers[num_of_HiddenLayers+1][i]);
              }
              verror=verror/2;
              error_val+=verror;

          }

          error_epoch /= Instances;
          cout<<error_epoch<<endl;

          error_val /= validation.size();
          cout<<error_val<<endl;
          cout<<endl;

          if (error_val<min_val_error)
          {
              min_val_error=error_val;
              best_w=weight;
              best_epoch=e;
              best_bias_w=bias_w;
              training_error_at_minvalerror=error_epoch;

          }
  }

    cout<<"----------------------------------------------------------------------"<<endl;
    cout<<"The minimum error of validation set is "<<min_val_error<<endl;
    cout<<"Error of training set at this point is "<<training_error_at_minvalerror<<endl;
    cout<<"----------------------------------------------------------------------"<<endl;

  return 0;
}

/*---------------------------------------------------------*/
void print_out(const vector<double> &v)
{
    int mi=0,m=v[0];
    for (int i=1; i<v.size(); i++)
    {
        if(m<v[i])
        {
            mi=i;
            m=v[i];
        }
    }
    vector<int>p(v.size(),0);
    p[mi]=1;
    for (int i=0; i<p.size(); i++)
    {
        cout<<p[i]<<" ";
    }cout<<endl;

}
/*---------------------------------------------------------*/
void classify(const vector<double>& data_output,vector<vector<double>> &target)
{
   int i;
   for(i=0;i<data_output.size();i++)
   {
       target[i][data_output[i]-1]=1;
   }
}
/*---------------------------------------------------------*/

double sigmoid(double x)
{
    double ret =  1/(1 + exp(-x));
    return ret;
}

double dSigmoid(double x)
{
    //double ret = sigmoid(x)*(1 - sigmoid(x));
    double ret = x*(1-x);
    return ret;
}
/*---------------------------------------------------------*/
double tanh(double x)
{
    double ret;
    ret = 2/(1 + exp(-2*x));
    return ret;
}

double dtanh(double x)
{
    //double ret = tanh(x) * tanh(x);
    double ret = x*x;
    return (1-ret);
}

/*---------------------------------------------------------*/
double ReLU(double x)
{
    double ret;
    if(x<0)
        ret = 0;
    else
        ret = x;

    return ret;
}

double dReLU(double x)
{
    double ret;
    if(x<0)
        ret=0;
    else
        ret=1;

    return ret;
}


double convert_to_num(string str)
{
   // object from the class stringstream
    stringstream geek(str);

    // The object has the value 12345 and stream
    // it to the integer x
    double x = 0;
    geek >> x;
    return x;
}


double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

//for feed forward
double Activation(int type,double value)
{

    double ret=0;
    switch(type)
    {
        case 1: {
                 ret = sigmoid(value);
                 break;
                }
        case 2: {
                  ret = ReLU(value);
                  break;
                }
        case 3:{
                  ret = tanh(value);
                  break;
               }
        default:{
                 cout<<"Invalid Activation ID"<<endl;
                 break;
                }
    }
    return ret;
}

//for backpropogation
double Activation_Derivative(int type,double value)
{
    double ret=0;
    switch(type)
    {
        case 1: {
                 ret = dSigmoid(value);
                 break;
                }
        case 2: {
                  ret = dReLU(value);
                  break;
                }
        case 3:{
                  ret = dtanh(value);
                  break;
               }
        default:{
                 cout<<"Invalid Activation ID"<<endl;
                 break;
                }
    }
    return ret;
}

void file_read(vector<vector<double> >&dataset)
{
    ifstream fin;
    fin.open("/Users/shikhagupta/Desktop/ml_assignment/ml_assignment/Newdata.txt");
    while(fin)
       {
           string str;
           getline(fin,str);
           if(str=="endl")
           {
               break;
           }
           vector<string> v;
           vector<double>row;
           stringstream ss(str);

           while (ss.good())
           {
               string substr;
               getline(ss, substr, ',');
               v.push_back(substr);
               row.push_back(convert_to_num(substr));
           }

           dataset.push_back(row);
      }
       fin.close();

}

//to split data into training and testing
void split_data( vector<vector<double>> &dataset,vector<vector<double>> &validation,vector<vector<double>> &target,vector<vector<double>> &validation_target)
{
        int train_data_size=80,val_data_size=20;
        //default values 80-20
        cout<<"enter training data percentage"<<endl;
        cin>>train_data_size;
        train_data_size=train_data_size*dataset.size()/100;
        val_data_size=dataset.size()-train_data_size;
        for (int i=0; i<val_data_size; i++)
        {
            validation.push_back(dataset.back());
            dataset.pop_back();
            validation_target.push_back(target.back());
            target.pop_back();
        }
}
