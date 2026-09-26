#include "subtree_crossover.h"
#include <stdexcept>
#include <memory>

// TreeNode implementation is in base_crossover.cc

// ============================================================================
// SUBTREE CROSSOVER IMPLEMENTATION
// ============================================================================

int SubtreeCrossover::countNodes(const TreeNode* tree) {
    if (!tree) return 0;
    int count = 1;
    for (const auto* child : tree->children) {
        count += countNodes(child);
    }
    return count;
}

TreeNode* SubtreeCrossover::getNodeAtIndex(TreeNode* tree, int index, int& current) {
    if (!tree) return nullptr;
    
    if (current == index) {
        return tree;
    }
    current++;
    
    for (auto* child : tree->children) {
        TreeNode* result = getNodeAtIndex(child, index, current);
        if (result) return result;
    }
    
    return nullptr;
}

TreeNode* SubtreeCrossover::selectRandomNode(TreeNode* tree) {
    if (!tree) return nullptr;
    
    int node_count = countNodes(tree);
    std::uniform_int_distribution<int> dist(0, node_count - 1);
    int target_index = dist(rng);
    
    int current = 0;
    return getNodeAtIndex(tree, target_index, current);
}

std::pair<TreeNode*, TreeNode*> SubtreeCrossover::crossover(const TreeNode* parent1, const TreeNode* parent2) {
    if (!parent1 || !parent2) {
        throw std::invalid_argument("Parents cannot be null");
    }
    
    operation_count++;
    
    TreeNode* child1 = parent1->clone();
    TreeNode* child2 = parent2->clone();
    
    TreeNode* node1 = selectRandomNode(child1);
    TreeNode* node2 = selectRandomNode(child2);
    
    // Swap the subtrees by swapping their values and children
    std::swap(node1->value, node2->value);
    std::swap(node1->children, node2->children);
    
    return {child1, child2};
}
