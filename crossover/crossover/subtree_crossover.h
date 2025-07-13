#ifndef SUBTREE_CROSSOVER_H
#define SUBTREE_CROSSOVER_H

#include "base_crossover.h"

// Tree node structure for GP
struct TreeNode {
    std::string value;
    std::vector<TreeNode*> children;
    
    TreeNode(const std::string& val) : value(val) {}
    ~TreeNode();
    TreeNode* clone() const;
};

class SubtreeCrossover : public CrossoverOperator {
public:
    SubtreeCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("SubtreeCrossover", seed) {}
    
    std::pair<TreeNode*, TreeNode*> crossover(const TreeNode* parent1, const TreeNode* parent2);
    
private:
    TreeNode* selectRandomNode(TreeNode* tree);
    int countNodes(const TreeNode* tree);
    TreeNode* getNodeAtIndex(TreeNode* tree, int index, int& current);
};

#endif // SUBTREE_CROSSOVER_H
