#include "base_crossover.h"
#include <algorithm>
#include <iostream>

// ============================================================================
// TREE NODE IMPLEMENTATION
// ============================================================================

TreeNode::~TreeNode() {
    for (auto* child : children) {
        delete child;
    }
}

TreeNode* TreeNode::clone() const {
    TreeNode* copy = new TreeNode(value);
    for (const auto* child : children) {
        copy->children.push_back(child->clone());
    }
    return copy;
}
