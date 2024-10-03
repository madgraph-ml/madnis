from dataclasses import dataclass


@dataclass
class ChannelGroup:
    """
    Args:
        group_index:
        target_index:
        channel_indices:
    """

    group_index: int
    target_index: int
    channel_indices: list[int]


@dataclass
class ChannelData:
    """
    Args:
        channel_index:
        target_index:
        group:
        remapped:
        position_in_group:
    """

    channel_index: int
    target_index: int
    group: ChannelGroup
    remapped: bool
    position_in_group: int


class ChannelGrouping:
    def __init__(self, channel_assignment: list[int | None]):
        """

        Args:
            channel_assignment:
        """
        group_dict = {}
        for source_channel, target_channel in enumerate(channel_assignment):
            if target_channel is None:
                group_dict[source_channel] = ChannelGroup(
                    group_index=len(group_dict),
                    target_index=source_channel,
                    channel_indices=[source_channel],
                )

        self.channels: list[ChannelData] = []
        self.groups: list[ChannelGroup] = list(group_dict.values())

        for source_channel, target_channel in enumerate(channel_assignment):
            if target_channel is None:
                self.channels.append(
                    ChannelData(
                        channel_index=source_channel,
                        target_index=source_channel,
                        group=group_dict[source_channel],
                        remapped=False,
                        position_in_group=0,
                    )
                )
            else:
                group = group_dict[target_channel]
                self.channels.append(
                    ChannelData(
                        channel_index=source_channel,
                        target_index=target_channel,
                        group=group,
                        remapped=True,
                        position_in_group=len(group.channel_indices),
                    )
                )
                group.channel_indices.append(source_channel)
