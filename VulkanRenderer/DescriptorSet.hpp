#pragma once

#include "RHI.hpp"

namespace Gfx
{
	class DescriptorSet
	{
	private:
		friend class RHI;

		DescriptorSet(const std::shared_ptr<vk::raii::DescriptorPool>& descriptorPool, vk::raii::DescriptorSet&& descriptorSet);

	public:
		DescriptorSet(nullptr_t) :
			m_descriptorPool(nullptr),
			m_descriptorSet(nullptr)
		{
		}

		DescriptorSet() = delete;

		operator vk::DescriptorSet() const { return *m_descriptorSet; }
		vk::DescriptorSet operator*() const { return *m_descriptorSet; }

	private:
		std::shared_ptr<vk::raii::DescriptorPool> m_descriptorPool;
		vk::raii::DescriptorSet m_descriptorSet;
	};
}