#include "Pipeline.hpp"

Gfx::Pipeline::Pipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& pipelineLayout, vk::raii::DescriptorSetLayout&& descriptorSetLayout):
	m_pipeline(std::move(pipeline)),
	m_pipelineLayout(std::move(pipelineLayout)),
	m_descriptorSetLayout(std::move(descriptorSetLayout))
{
}
