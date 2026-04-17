#pragma once

#include "RHI.hpp"

namespace Gfx
{
	class Pipeline
	{
	private:
		friend class RHI;

		Pipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& pipelineLayout, vk::raii::DescriptorSetLayout&& descriptorSetLayout);

	public:
		Pipeline(nullptr_t) :
			m_pipeline(nullptr),
			m_pipelineLayout(nullptr),
			m_descriptorSetLayout(nullptr)
		{
		}

		Pipeline() = delete;

		operator vk::Pipeline() const { return *m_pipeline; }
		vk::Pipeline operator*() const { return *m_pipeline; }

		const vk::raii::PipelineLayout& getPipelineLayout() const { return m_pipelineLayout; }
		const vk::raii::DescriptorSetLayout& getDescriptorSetLayout() const { return m_descriptorSetLayout; }

	private:
		vk::raii::Pipeline m_pipeline;
		vk::raii::PipelineLayout m_pipelineLayout;
		vk::raii::DescriptorSetLayout m_descriptorSetLayout;
	};
}
