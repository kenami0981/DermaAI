using Derma.Domain;

namespace Derma.Application.Core
{
    public interface ITokenService
    {
        string CreateToken(AppUser user);
    }
}