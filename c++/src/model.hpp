#pragma once

#include "common.hpp"
#include "corpus.hpp"

enum action_t { COPY, INIT, FREE };

class model
{
public:
    model(corpus* corp) : corp(corp)
    {
        nUsers = corp->nUsers;
        nItems = corp->nItems;
        nVotes = corp->nVotes;
        
        // split into training set AND valid set AND test set with ratio of 3:1:1
        test_per_user = new map<int,long long>[nUsers];
        val_per_user  = new map<int,long long>[nUsers];
        pos_per_user = new map<int,long long>[nUsers];
        pos_per_item = new map<int,long long>[nItems];
        for (int x = 0; x < nVotes; x ++) {
            vote* v = corp->V.at(x);
            int user = v->user;
            int item = v->item;
            long long voteTime = v->voteTime; 
        
            if (x < nVotes * 0.2) { // add to test set
                test_per_user[user][item] = voteTime;
            } else if (x < nVotes * 0.4 && x >= nVotes * 0.2) { // add to validation set
                val_per_user[user][item] = voteTime;
            }
            else {// add to training set
                pos_per_user[user][item] = voteTime;
                pos_per_item[item][user] = voteTime;
            }
        }
        
        // calculate num_pos_events
        num_pos_events = 0;
        for (int u = 0; u < nUsers; u ++) {
            num_pos_events += pos_per_user[u].size();
        }
    }

    ~model()
    {
        delete [] pos_per_user;
        delete [] pos_per_item;
    
        delete [] test_per_user;
        delete [] val_per_user;
    }
    
    /* Model parameters */
    int NW; // Total number of parameters
    double* W; // Contiguous version of all parameters
    double* bestW;
    
    /* Corpus related */
    corpus* corp; // dangerous
    int nUsers; // Number of users
    int nItems; // Number of items
    int nVotes; // Number of ratings
    
    map<int,long long>* pos_per_user;
    map<int,long long>* pos_per_item;
    
    map<int,long long>* val_per_user;
    map<int,long long>* test_per_user;
    
    int num_pos_events;
    
    virtual double precision(int);
    virtual double precision_coldItem(int);
    virtual double recall(int);
    virtual double recall_coldItem(int);

private:
    virtual double prediction(int user, int item) = 0;
};
