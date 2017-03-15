#include "model.hpp"

bool compare_pair(const pair<int, double> & m1, const pair<int, double> & m2);

double model::precision(int M)
{
    vector<pair<int, double>> *rates = new vector<pair<int, double>>[nUsers];
    double Precision[nUsers];
    
    for(int i = 0; i < nUsers; i ++)
    {
        for(int j = 0; j < nItems; j ++)
        {
            if(pos_per_user[i].find(j) != pos_per_user[i].end())
                continue;
            if(val_per_user[i].find(j) != val_per_user[i].end())
                continue;
            rates[i].push_back(pair<int, double>(j, prediction(i, j)));
        }
        partial_sort(rates[i].begin(), rates[i].begin() + M, rates[i].end(), compare_pair);
        int nHit = 0;
        for(int tk = 0; tk < M; tk ++)
        {
            if(test_per_user[i].find(rates[i][tk].first) != test_per_user[i].end()) nHit ++;
        }
        Precision[i] = float(nHit) / M;
    }
    
    double sum = 0;
    for(int i = 0; i < nUsers; i ++)
    {
        sum += Precision[i];
    }
    
    return sum / nUsers;
    return 0;
}

double model::precision_coldItem(int M)
{
    vector<pair<int, double>> *rates = new vector<pair<int, double>>[nUsers];
    double Precision[nUsers];
    
    for(int i = 0; i < nUsers; i ++)
    {
        for(int j = 0; j < nItems; j ++)
        {
            if(pos_per_item[j].size() > 5)
                continue;
            if(pos_per_user[i].find(j) != pos_per_user[i].end())
                continue;
            if(val_per_user[i].find(j) != val_per_user[i].end())
                continue;
            rates[i].push_back(pair<int, double>(j, prediction(i, j)));
        }
        partial_sort(rates[i].begin(), rates[i].begin() + M, rates[i].end(), compare_pair);
        int nHit = 0;
        for(int tk = 0; tk < M; tk ++)
        {
            if(test_per_user[i].find(rates[i][tk].first) != test_per_user[i].end()) nHit ++;
        }
        Precision[i] = float(nHit) / M;
    }
    
    double sum = 0;
    for(int i = 0; i < nUsers; i ++)
    {
        sum += Precision[i];
    }
    
    return sum / nUsers;
}

double model::recall(int M)
{
    vector<pair<int, double>> *rates = new vector<pair<int, double>>[nUsers];
    double Recall[nUsers];
    
    for(int i = 0; i < nUsers; i ++)
    {
        for(int j = 0; j < nItems; j ++)
        {
            if(pos_per_user[i].find(j) != pos_per_user[i].end())
                continue;
            if(val_per_user[i].find(j) != val_per_user[i].end())
                continue;
            rates[i].push_back(pair<int, double>(j, prediction(i, j)));
        }
        partial_sort(rates[i].begin(), rates[i].begin() + M, rates[i].end(), compare_pair);
        int nHit = 0;
        for(int tk = 0; tk < M; tk ++)
        {
            if(test_per_user[i].find(rates[i][tk].first) != test_per_user[i].end()) nHit ++;
        }
        if(test_per_user[i].size() == 0)
        {
            Recall[i] = 1;
        }
        else
        {
            Recall[i] = float(nHit) / test_per_user[i].size();
        }
    }
    
    double sum = 0;
    for(int i = 0; i < nUsers; i ++)
    {
        sum += Recall[i];
    }
    return sum / nUsers;
}

double model::recall_coldItem(int M)
{
    vector<pair<int, double>> *rates = new vector<pair<int, double>>[nUsers];
    double Recall[nUsers];
    
    for(int i = 0; i < nUsers; i ++)
    {
        for(int j = 0; j < nItems; j ++)
        {
            if(pos_per_item[j].size() > 5)
                continue;
            if(pos_per_user[i].find(j) != pos_per_user[i].end())
                continue;
            if(val_per_user[i].find(j) != val_per_user[i].end())
                continue;
            rates[i].push_back(pair<int, double>(j, prediction(i, j)));
        }
        partial_sort(rates[i].begin(), rates[i].begin() + M, rates[i].end(), compare_pair);
        int nHit = 0;
        for(int tk = 0; tk < M; tk ++)
        {
            if(test_per_user[i].find(rates[i][tk].first) != test_per_user[i].end()) nHit ++;
        }
        int nColdItem = 0;
        for(map<int,long long>::iterator ct = test_per_user[i].begin(); ct != test_per_user[i].end(); ct ++)
        {
            if(pos_per_item[ct->first].size() <= 5) nColdItem ++; 
        }
        Recall[i] = float(nHit) / nColdItem;
        if(test_per_user[i].size() == 0)
        {
            Recall[i] = 1;
        }
        else
        {
            Recall[i] = float(nHit) / nColdItem;
        }
    }
    
    double sum = 0;
    for(int i = 0; i < nUsers; i ++)
    {
        sum += Recall[i];
    }
    
    return sum / nUsers;
}

bool compare_pair(const pair<int, double> & m1, const pair<int, double> & m2)
{
    return m1.second < m2.second;
}
