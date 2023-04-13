#include <vk_initializers.h>
#include <vulkan/vulkan_core.h>


namespace vkinit {

VkCommandPoolCreateInfo command_pool_create_info(uint32_t queue_family_index, VkCommandPoolCreateFlags flags) {
  VkCommandPoolCreateInfo command_pool_info = {};

  command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_info.pNext = nullptr;
  command_pool_info.queueFamilyIndex = queue_family_index;
  command_pool_info.flags = flags;

  return command_pool_info;
}

VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count, VkCommandBufferLevel level) {
  VkCommandBufferAllocateInfo command_alloc_info = {};

  command_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_alloc_info.pNext = nullptr;
  command_alloc_info.commandPool = pool;
  command_alloc_info.commandBufferCount = 1;
  command_alloc_info.level = level;

  return command_alloc_info;
}

} // vkinit
