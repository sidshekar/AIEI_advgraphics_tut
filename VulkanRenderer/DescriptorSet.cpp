#include "DescriptorSet.hpp"

using Gfx::DescriptorSet;

DescriptorSet::DescriptorSet(const std::shared_ptr<vk::raii::DescriptorPool>& descriptorPool, vk::raii::DescriptorSet&& descriptorSet) :
	m_descriptorPool(descriptorPool),
	m_descriptorSet(std::move(descriptorSet))
{
}