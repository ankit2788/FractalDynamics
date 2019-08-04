#include <vector>
#include <iostream>

std::vector<int> funNonZero(std::vector<int> &myVector)
{
    
    std::vector<int>::iterator it = myVector.begin();

    while (it != myVector.end())
    {

        if (*it == 0)
        {
            it = myVector.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return myVector;

    /*
    for (std::vector<int>::iterator it = myVector.begin(); it != myVector.end(); it++)
    {
        if (*it != 0)
        {

        }

    }*/

}

int main()
{
    std::vector<int> myvector;
    myvector.push_back(7);
    myvector.push_back(0);
    myvector.push_back(0);
    myvector.push_back(6);
    myvector.push_back(4);
    myvector.push_back(9);
    myvector.push_back(0);

    for (int i=0;i<myvector.size();++i)
    {
        std::cout<<myvector[i]<<" ";
    }

    std::cout<<"\n"<<"after deleting elements:"<<"\n";
    funNonZero(myvector);

    

    for (int i=0;i<myvector.size();++i)
    {
        std::cout<<myvector[i]<<" ";
    }
    
    std::cout<<"\n";

    return 0;
}