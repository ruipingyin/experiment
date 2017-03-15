#pragma once

#include "common.hpp"

class corpus
{
public:
    corpus() {}
    ~corpus() {}

    vector<vote*> V; // vote

    int nUsers; // Number of users
    int nItems; // Number of items
    int nVotes; // Number of ratings

    map<string, int> userIds; // Maps a user's string-valued ID to an integer
    map<string, int> itemIds; // Maps an item's string-valued ID to an integer

    map<int, string> rUserIds; // Inverse of the above maps
    map<int, string> rItemIds;

    /* For pre-load */
    map<string, int> uCounts;
    map<string, int> bCounts;

    vector<vector<pair<int, float> > > imageFeatures;
    int imFeatureDim;  // fixed to 4096

    void loadData(const char* voteFile, int userMin, int itemMin);
    void save2Txt();
    void cleanUp();

    private:
    void loadVotes(const char* voteFile, int userMin, int itemMin);
    void generateVotes(map<pair<int, int>, long long>& voteMap);
};
